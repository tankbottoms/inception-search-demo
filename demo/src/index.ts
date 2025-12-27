/**
 * Inception Demo Client
 *
 * Commands:
 *   demo      - Run full demo (extract PDF text -> embed -> search)
 *   ocr       - Process PDFs with OCR (PDF -> images -> OCR -> text)
 *   pipeline  - Full OCR pipeline (PDF -> OCR -> embed -> search)
 *   index     - Index documents to output/
 *   search    - Search indexed documents
 *   benchmark - Run indexing + search benchmarks
 */

import { Command } from 'commander';
import axios, { AxiosError } from 'axios';
import chalk from 'chalk';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { join, basename, extname } from 'path';
import sharp from 'sharp';

// PDF parsing - dynamic import for ESM compatibility
let pdfParse: typeof import('pdf-parse');

const API_URL = process.env.API_URL || 'http://localhost:8005';
const FILES_DIR = join(import.meta.dir, '../files');
const OUTPUT_DIR = join(import.meta.dir, '../output');
const IMAGES_DIR = join(OUTPUT_DIR, 'images');

// ============================================================
// Types
// ============================================================

interface ChunkEmbedding {
  chunk_number: number;
  chunk: string;
  embedding: number[];
}

interface TextResponse {
  id: number;
  embeddings: ChunkEmbedding[];
}

interface IndexedDocument {
  id: number;
  filename: string;
  originalName: string;
  chunks: ChunkEmbedding[];
  textLength: number;
  extractedAt: string;
  source: 'text' | 'ocr';
  pageCount?: number;
}

interface SearchResult {
  documentId: number;
  filename: string;
  originalName: string;
  chunkNumber: number;
  score: number;
  text: string;
}

interface BenchmarkResult {
  operation: string;
  count: number;
  totalMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
}

interface OCRResult {
  text: string;
  pages: Array<{
    page: number;
    text: string;
    confidence?: number;
  }>;
  timing: {
    total_ms: number;
    api_ms?: number;
  };
  provider: string;
}

interface OCRProviderStatus {
  hunyuan: { available: boolean; details?: string };
  mistral: { available: boolean; details?: string };
  auto: { available: boolean; details?: string };
}

// ============================================================
// Utilities
// ============================================================

async function initPdfParse() {
  if (!pdfParse) {
    pdfParse = (await import('pdf-parse')).default;
  }
  return pdfParse;
}

function loadFilenameMapping(): Record<string, string> {
  const mappingPath = join(FILES_DIR, 'filename-mapping.json');
  if (existsSync(mappingPath)) {
    return JSON.parse(readFileSync(mappingPath, 'utf-8'));
  }
  return {};
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// Use cyan instead of gray for better visibility on dark terminals (e.g., Solarized Dark)
const dim = chalk.cyan;

async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${API_URL}/health`, { timeout: 5000 });
    return response.data?.status === 'ok';
  } catch {
    return false;
  }
}

async function getOCRProviders(): Promise<OCRProviderStatus> {
  try {
    const response = await axios.get(`${API_URL}/api/v1/ocr/providers`, { timeout: 5000 });
    return response.data;
  } catch {
    return {
      hunyuan: { available: false, details: 'API unreachable' },
      mistral: { available: false, details: 'API unreachable' },
      auto: { available: false, details: 'API unreachable' },
    };
  }
}

// ============================================================
// PDF Processing
// ============================================================

async function extractPdfText(pdfPath: string): Promise<string> {
  const parse = await initPdfParse();
  const buffer = readFileSync(pdfPath);
  const result = await parse(buffer);
  return result.text;
}

async function getPdfPageCount(pdfPath: string): Promise<number> {
  const parse = await initPdfParse();
  const buffer = readFileSync(pdfPath);
  const result = await parse(buffer);
  return result.numpages;
}

/**
 * Convert PDF pages to images using pdf2pic
 * Requires graphicsmagick or imagemagick installed
 */
async function pdfToImages(pdfPath: string, outputDir: string): Promise<string[]> {
  const { fromPath } = await import('pdf2pic');

  const filename = basename(pdfPath, '.pdf');
  const imageDir = join(outputDir, filename);
  mkdirSync(imageDir, { recursive: true });

  const options = {
    density: 150,           // DPI
    saveFilename: 'page',
    savePath: imageDir,
    format: 'png',
    width: 1200,
    height: 1600,
  };

  const convert = fromPath(pdfPath, options);
  const pageCount = await getPdfPageCount(pdfPath);

  const imagePaths: string[] = [];

  for (let i = 1; i <= pageCount; i++) {
    try {
      const result = await convert(i);
      if (result.path) {
        imagePaths.push(result.path);
      }
    } catch (error) {
      console.log(chalk.yellow(`    Warning: Could not convert page ${i}`));
    }
  }

  return imagePaths;
}

/**
 * Alternative: Use sharp directly for single-page PDFs or images
 */
async function convertImageForOCR(imagePath: string): Promise<Buffer> {
  const image = await sharp(imagePath)
    .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer();

  return image;
}

// ============================================================
// API Calls
// ============================================================

async function embedDocument(id: number, text: string): Promise<TextResponse> {
  const response = await axios.post<TextResponse>(
    `${API_URL}/api/v1/embed/text`,
    { id, text },
    { timeout: 120000 }
  );
  return response.data;
}

async function embedQuery(text: string): Promise<number[]> {
  const response = await axios.post<{ embedding: number[] }>(
    `${API_URL}/api/v1/embed/query`,
    { text },
    { timeout: 30000 }
  );
  return response.data.embedding;
}

async function performOCR(
  imageBuffer: Buffer,
  options: { provider?: string; maxTokens?: number } = {}
): Promise<OCRResult> {
  const base64Image = imageBuffer.toString('base64');
  const dataUri = `data:image/png;base64,${base64Image}`;

  const response = await axios.post<OCRResult>(
    `${API_URL}/api/v1/ocr`,
    {
      document: dataUri,
      provider: options.provider || 'auto',
      maxTokens: options.maxTokens || 4096,
    },
    { timeout: 300000 } // 5 minute timeout for OCR
  );

  return response.data;
}

async function performOCRBatch(
  imageBuffers: Buffer[],
  options: { provider?: string; maxTokens?: number } = {}
): Promise<OCRResult> {
  const documents = imageBuffers.map(buf => {
    const base64Image = buf.toString('base64');
    return `data:image/png;base64,${base64Image}`;
  });

  const response = await axios.post<OCRResult>(
    `${API_URL}/api/v1/ocr/batch`,
    {
      documents,
      provider: options.provider || 'auto',
      maxTokens: options.maxTokens || 4096,
    },
    { timeout: 600000 } // 10 minute timeout for batch
  );

  return response.data;
}

// ============================================================
// Index Management
// ============================================================

function loadIndex(): IndexedDocument[] {
  const indexPath = join(OUTPUT_DIR, 'index.json');
  if (existsSync(indexPath)) {
    return JSON.parse(readFileSync(indexPath, 'utf-8'));
  }
  return [];
}

function saveIndex(docs: IndexedDocument[]): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const indexPath = join(OUTPUT_DIR, 'index.json');
  writeFileSync(indexPath, JSON.stringify(docs, null, 2));
}

// ============================================================
// Commands
// ============================================================

const program = new Command();

program
  .name('inception-demo')
  .description('Demo client for Inception ONNX inference service')
  .version('2.0.0');

/**
 * Demo Command - Full pipeline with text extraction
 */
program
  .command('demo')
  .description('Run full demo (extract PDF text -> embed -> search)')
  .option('--pdf-count <n>', 'Number of PDFs to process (0 = all)', '3')
  .option('--query <text>', 'Search query after indexing', 'Jan Wallace complaint')
  .action(async (options) => {
    console.log(chalk.blue('\n--- Inception ONNX Demo ---\n'));

    // Check API health
    console.log(dim(`Checking API at ${API_URL}...`));
    const healthy = await checkApiHealth();
    if (!healthy) {
      console.log(chalk.red('X API is not available. Start the backend first:'));
      console.log(dim('  $ bun run dev'));
      process.exit(1);
    }
    console.log(chalk.green('OK API is healthy\n'));

    // Get PDF files
    const mapping = loadFilenameMapping();
    let pdfFiles = readdirSync(FILES_DIR)
      .filter(f => f.endsWith('.pdf'))
      .slice(0, options.pdfCount === '0' ? undefined : parseInt(options.pdfCount));

    console.log(chalk.white.bold(`Processing ${pdfFiles.length} PDF files:\n`));

    const documents: IndexedDocument[] = [];
    let totalTime = 0;

    for (let i = 0; i < pdfFiles.length; i++) {
      const filename = pdfFiles[i];
      const originalName = mapping[filename] || filename;
      const pdfPath = join(FILES_DIR, filename);

      console.log(chalk.white(`${i + 1}. ${filename}`));
      console.log(dim(`   Original: ${originalName}`));

      const startTime = performance.now();

      // Extract text
      console.log(dim('   Extracting text...'));
      let text: string;
      try {
        text = await extractPdfText(pdfPath);
        console.log(dim(`   Extracted ${text.length} characters`));
      } catch (error) {
        console.log(chalk.yellow(`   Warning: Text extraction failed, skipping`));
        continue;
      }

      if (text.trim().length < 100) {
        console.log(chalk.yellow(`   Warning: Insufficient text content (may need OCR), skipping`));
        continue;
      }

      // Generate embeddings
      console.log(dim('   Generating embeddings...'));
      try {
        const result = await embedDocument(i + 1, text);
        const elapsed = performance.now() - startTime;
        totalTime += elapsed;

        documents.push({
          id: i + 1,
          filename,
          originalName,
          chunks: result.embeddings,
          textLength: text.length,
          extractedAt: new Date().toISOString(),
          source: 'text',
        });

        console.log(chalk.green(`   OK ${result.embeddings.length} chunks in ${formatTime(elapsed)}`));
      } catch (error) {
        const errMsg = error instanceof AxiosError ? error.response?.data?.error || error.message : String(error);
        console.log(chalk.red(`   X Embedding failed: ${errMsg}`));
      }

      console.log('');
    }

    // Save index
    saveIndex(documents);
    console.log(chalk.blue('--- Indexing Complete ---\n'));
    console.log(dim(`Documents: ${documents.length}`));
    console.log(dim(`Total chunks: ${documents.reduce((sum, d) => sum + d.chunks.length, 0)}`));
    console.log(dim(`Total time: ${formatTime(totalTime)}`));
    console.log(dim(`Index saved to: ${OUTPUT_DIR}/index.json`));
    console.log('');

    // Search demo
    if (documents.length > 0) {
      console.log(chalk.blue('--- Search Demo ---\n'));
      console.log(chalk.white(`Query: "${options.query}"\n`));

      const queryStart = performance.now();
      const queryEmbedding = await embedQuery(options.query);
      const queryTime = performance.now() - queryStart;

      // Calculate similarities
      const results: SearchResult[] = [];
      for (const doc of documents) {
        for (const chunk of doc.chunks) {
          const score = cosineSimilarity(queryEmbedding, chunk.embedding);
          results.push({
            documentId: doc.id,
            filename: doc.filename,
            originalName: doc.originalName,
            chunkNumber: chunk.chunk_number,
            score,
            text: chunk.chunk,
          });
        }
      }

      // Sort by score descending
      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, 5);

      console.log(dim(`Query embedding: ${queryTime.toFixed(1)}ms\n`));
      console.log(chalk.white.bold('Top 5 Results:'));

      for (let i = 0; i < topResults.length; i++) {
        const r = topResults[i];
        console.log(chalk.white(`\n${i + 1}. Score: ${(r.score * 100).toFixed(1)}%`));
        console.log(dim(`   Document: ${r.originalName}`));
        console.log(dim(`   Chunk: ${r.chunkNumber}`));
        console.log(dim(`   Preview: ${r.text.slice(0, 150)}...`));
      }

      console.log('');
    }
  });

/**
 * OCR Command - Process PDFs with OCR
 */
program
  .command('ocr')
  .description('Process PDFs with OCR (requires graphicsmagick for PDF conversion)')
  .option('--pdf <path>', 'Single PDF file to process')
  .option('--image <path>', 'Single image file to process')
  .option('--provider <name>', 'OCR provider (hunyuan, mistral, auto)', 'auto')
  .option('--output <path>', 'Output file for extracted text')
  .action(async (options) => {
    console.log(chalk.blue('\n--- OCR Processing ---\n'));

    // Check API and OCR providers
    console.log(dim(`Checking API at ${API_URL}...`));
    const healthy = await checkApiHealth();
    if (!healthy) {
      console.log(chalk.red('X API is not available. Start the backend first.'));
      process.exit(1);
    }

    const providers = await getOCRProviders();
    console.log(chalk.white.bold('OCR Providers:'));
    console.log(dim(`  Hunyuan: ${providers.hunyuan.available ? chalk.green('Available') : chalk.red('Unavailable')} - ${providers.hunyuan.details}`));
    console.log(dim(`  Mistral: ${providers.mistral.available ? chalk.green('Available') : chalk.red('Unavailable')} - ${providers.mistral.details}`));
    console.log('');

    if (!providers.auto.available) {
      console.log(chalk.red('X No OCR providers available.'));
      console.log(dim('  Install HunyuanOCR model or set MISTRAL_API_KEY'));
      process.exit(1);
    }

    let imageBuffer: Buffer;
    let sourceName: string;

    if (options.image) {
      // Process single image
      if (!existsSync(options.image)) {
        console.log(chalk.red(`X Image not found: ${options.image}`));
        process.exit(1);
      }
      console.log(chalk.white(`Processing image: ${options.image}`));
      imageBuffer = await convertImageForOCR(options.image);
      sourceName = basename(options.image);

    } else if (options.pdf) {
      // Process PDF - convert first page to image
      if (!existsSync(options.pdf)) {
        console.log(chalk.red(`X PDF not found: ${options.pdf}`));
        process.exit(1);
      }

      console.log(chalk.white(`Processing PDF: ${options.pdf}`));
      console.log(dim('  Converting to images...'));

      mkdirSync(IMAGES_DIR, { recursive: true });

      try {
        const imagePaths = await pdfToImages(options.pdf, IMAGES_DIR);
        if (imagePaths.length === 0) {
          console.log(chalk.red('X No pages could be converted'));
          process.exit(1);
        }

        console.log(chalk.green(`  OK Converted ${imagePaths.length} pages`));

        // Process first page for now
        imageBuffer = await convertImageForOCR(imagePaths[0]);
        sourceName = basename(options.pdf);

      } catch (error) {
        console.log(chalk.red(`X PDF conversion failed: ${error}`));
        console.log(dim('  Make sure graphicsmagick or imagemagick is installed'));
        process.exit(1);
      }

    } else {
      console.log(chalk.yellow('Provide --pdf or --image option'));
      process.exit(1);
    }

    // Perform OCR
    console.log(dim(`  Running OCR with provider: ${options.provider}...`));
    const startTime = performance.now();

    try {
      const result = await performOCR(imageBuffer, { provider: options.provider });
      const elapsed = performance.now() - startTime;

      console.log(chalk.green(`\nOK OCR Complete in ${formatTime(elapsed)}`));
      console.log(dim(`Provider: ${result.provider}`));
      console.log(dim(`Characters extracted: ${result.text.length}`));

      console.log(chalk.white.bold('\nExtracted Text:'));
      console.log(chalk.white('─'.repeat(60)));
      console.log(result.text.slice(0, 2000));
      if (result.text.length > 2000) {
        console.log(dim(`\n... (${result.text.length - 2000} more characters)`));
      }
      console.log(chalk.white('─'.repeat(60)));

      // Save output if requested
      if (options.output) {
        writeFileSync(options.output, result.text);
        console.log(dim(`\nSaved to: ${options.output}`));
      }

    } catch (error) {
      const errMsg = error instanceof AxiosError ? error.response?.data?.error || error.message : String(error);
      console.log(chalk.red(`X OCR failed: ${errMsg}`));
      process.exit(1);
    }
  });

/**
 * Pipeline Command - Full OCR -> Embed -> Search pipeline
 */
program
  .command('pipeline')
  .description('Full pipeline: PDF -> OCR -> Embed -> Search')
  .option('--pdf-count <n>', 'Number of PDFs to process', '3')
  .option('--provider <name>', 'OCR provider (hunyuan, mistral, auto)', 'auto')
  .option('--query <text>', 'Search query after indexing', 'legal complaint')
  .option('--force-ocr', 'Use OCR even if text extraction works', false)
  .action(async (options) => {
    console.log(chalk.blue('\n--- Full OCR Pipeline ---\n'));

    // Check API
    const healthy = await checkApiHealth();
    if (!healthy) {
      console.log(chalk.red('X API is not available'));
      process.exit(1);
    }
    console.log(chalk.green('OK API is healthy'));

    // Check OCR providers
    const providers = await getOCRProviders();
    if (!providers.auto.available) {
      console.log(chalk.red('X No OCR providers available'));
      process.exit(1);
    }
    console.log(chalk.green(`OK OCR available (${providers.auto.details})\n`));

    // Get PDF files
    const mapping = loadFilenameMapping();
    let pdfFiles = readdirSync(FILES_DIR)
      .filter(f => f.endsWith('.pdf'))
      .slice(0, parseInt(options.pdfCount));

    console.log(chalk.white.bold(`Processing ${pdfFiles.length} PDFs with OCR pipeline:\n`));

    mkdirSync(IMAGES_DIR, { recursive: true });

    const documents: IndexedDocument[] = [];
    let totalTime = 0;

    for (let i = 0; i < pdfFiles.length; i++) {
      const filename = pdfFiles[i];
      const originalName = mapping[filename] || filename;
      const pdfPath = join(FILES_DIR, filename);

      console.log(chalk.white(`${i + 1}. ${filename}`));

      const startTime = performance.now();
      let text = '';
      let useOCR = options.forceOcr;
      let pageCount = 0;

      // Try text extraction first
      if (!useOCR) {
        try {
          text = await extractPdfText(pdfPath);
          pageCount = await getPdfPageCount(pdfPath);

          if (text.trim().length < 100) {
            console.log(dim('   Text extraction insufficient, using OCR'));
            useOCR = true;
          } else {
            console.log(dim(`   Text extracted: ${text.length} chars from ${pageCount} pages`));
          }
        } catch {
          console.log(dim('   Text extraction failed, using OCR'));
          useOCR = true;
        }
      }

      // Use OCR if needed
      if (useOCR) {
        console.log(dim('   Converting PDF to images...'));

        try {
          const imagePaths = await pdfToImages(pdfPath, IMAGES_DIR);
          pageCount = imagePaths.length;

          if (imagePaths.length === 0) {
            console.log(chalk.yellow('   Warning: No pages converted, skipping'));
            continue;
          }

          console.log(dim(`   Running OCR on ${imagePaths.length} pages...`));

          // Process each page with OCR
          const pageTexts: string[] = [];

          for (let p = 0; p < imagePaths.length; p++) {
            const imageBuffer = await convertImageForOCR(imagePaths[p]);
            const ocrResult = await performOCR(imageBuffer, { provider: options.provider });
            pageTexts.push(ocrResult.text);

            // Clean up image file
            try { unlinkSync(imagePaths[p]); } catch {}
          }

          text = pageTexts.join('\n\n--- Page Break ---\n\n');
          console.log(dim(`   OCR complete: ${text.length} chars`));

        } catch (error) {
          console.log(chalk.yellow(`   Warning: OCR failed: ${error}`));
          continue;
        }
      }

      // Generate embeddings
      if (text.trim().length < 50) {
        console.log(chalk.yellow('   Warning: Insufficient text, skipping'));
        continue;
      }

      console.log(dim('   Generating embeddings...'));

      try {
        const result = await embedDocument(i + 1, text);
        const elapsed = performance.now() - startTime;
        totalTime += elapsed;

        documents.push({
          id: i + 1,
          filename,
          originalName,
          chunks: result.embeddings,
          textLength: text.length,
          extractedAt: new Date().toISOString(),
          source: useOCR ? 'ocr' : 'text',
          pageCount,
        });

        console.log(chalk.green(`   OK ${result.embeddings.length} chunks in ${formatTime(elapsed)}`));

      } catch (error) {
        const errMsg = error instanceof AxiosError ? error.response?.data?.error || error.message : String(error);
        console.log(chalk.red(`   X Embedding failed: ${errMsg}`));
      }

      console.log('');
    }

    // Save index
    saveIndex(documents);

    const ocrDocs = documents.filter(d => d.source === 'ocr').length;
    const textDocs = documents.filter(d => d.source === 'text').length;

    console.log(chalk.blue('--- Pipeline Complete ---\n'));
    console.log(dim(`Documents indexed: ${documents.length}`));
    console.log(dim(`  - From text extraction: ${textDocs}`));
    console.log(dim(`  - From OCR: ${ocrDocs}`));
    console.log(dim(`Total chunks: ${documents.reduce((sum, d) => sum + d.chunks.length, 0)}`));
    console.log(dim(`Total time: ${formatTime(totalTime)}`));
    console.log('');

    // Run search
    if (documents.length > 0) {
      console.log(chalk.blue('--- Search Results ---\n'));
      console.log(chalk.white(`Query: "${options.query}"\n`));

      const queryEmbedding = await embedQuery(options.query);

      const results: SearchResult[] = [];
      for (const doc of documents) {
        for (const chunk of doc.chunks) {
          const score = cosineSimilarity(queryEmbedding, chunk.embedding);
          results.push({
            documentId: doc.id,
            filename: doc.filename,
            originalName: doc.originalName,
            chunkNumber: chunk.chunk_number,
            score,
            text: chunk.chunk,
          });
        }
      }

      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, 5);

      for (let i = 0; i < topResults.length; i++) {
        const r = topResults[i];
        const scoreColor = r.score > 0.7 ? chalk.green : r.score > 0.5 ? chalk.yellow : dim;
        console.log(`${i + 1}. ${scoreColor(`${(r.score * 100).toFixed(1)}%`)} - ${r.originalName}`);
        console.log(dim(`   Chunk ${r.chunkNumber}: ${r.text.slice(0, 100)}...`));
        console.log('');
      }
    }
  });

/**
 * Index Command
 */
program
  .command('index')
  .description('Index all PDFs in files/ directory')
  .option('--limit <n>', 'Maximum documents to index', '0')
  .option('--force', 'Re-index even if already indexed', false)
  .action(async (options) => {
    console.log(chalk.blue('\n--- Indexing Documents ---\n'));

    // Check API
    if (!await checkApiHealth()) {
      console.log(chalk.red('X API is not available'));
      process.exit(1);
    }

    const mapping = loadFilenameMapping();
    const limit = parseInt(options.limit);
    let pdfFiles = readdirSync(FILES_DIR).filter(f => f.endsWith('.pdf'));

    if (limit > 0) {
      pdfFiles = pdfFiles.slice(0, limit);
    }

    // Load existing index if not forcing
    let existingIndex = options.force ? [] : loadIndex();
    const indexedFilenames = new Set(existingIndex.map(d => d.filename));

    const documents: IndexedDocument[] = [...existingIndex];
    let processedCount = 0;
    let skippedCount = 0;

    for (const filename of pdfFiles) {
      if (indexedFilenames.has(filename)) {
        console.log(dim(`Skipping (already indexed): ${filename}`));
        skippedCount++;
        continue;
      }

      const originalName = mapping[filename] || filename;
      const pdfPath = join(FILES_DIR, filename);

      console.log(chalk.white(`Indexing: ${filename}`));

      try {
        const text = await extractPdfText(pdfPath);
        if (text.trim().length < 100) {
          console.log(chalk.yellow(`  Warning: Insufficient text, skipping`));
          continue;
        }

        const docId = documents.length + 1;
        const result = await embedDocument(docId, text);

        documents.push({
          id: docId,
          filename,
          originalName,
          chunks: result.embeddings,
          textLength: text.length,
          extractedAt: new Date().toISOString(),
          source: 'text',
        });

        console.log(chalk.green(`  OK ${result.embeddings.length} chunks`));
        processedCount++;
      } catch (error) {
        console.log(chalk.red(`  X Failed: ${error}`));
      }
    }

    saveIndex(documents);
    console.log(chalk.blue('\n--- Summary ---'));
    console.log(dim(`Processed: ${processedCount}`));
    console.log(dim(`Skipped: ${skippedCount}`));
    console.log(dim(`Total indexed: ${documents.length}`));
    console.log('');
  });

/**
 * Search Command
 */
program
  .command('search <query>')
  .description('Search indexed documents')
  .option('--limit <n>', 'Number of results', '10')
  .action(async (query, options) => {
    console.log(chalk.blue('\n--- Search ---\n'));

    const index = loadIndex();
    if (index.length === 0) {
      console.log(chalk.yellow('No documents indexed. Run "index" or "pipeline" first.'));
      process.exit(1);
    }

    // Check API
    if (!await checkApiHealth()) {
      console.log(chalk.red('X API is not available'));
      process.exit(1);
    }

    console.log(chalk.white(`Query: "${query}"`));
    console.log(dim(`Searching ${index.length} documents...\n`));

    const startTime = performance.now();
    const queryEmbedding = await embedQuery(query);
    const queryTime = performance.now() - startTime;

    // Calculate similarities
    const results: SearchResult[] = [];
    for (const doc of index) {
      for (const chunk of doc.chunks) {
        const score = cosineSimilarity(queryEmbedding, chunk.embedding);
        results.push({
          documentId: doc.id,
          filename: doc.filename,
          originalName: doc.originalName,
          chunkNumber: chunk.chunk_number,
          score,
          text: chunk.chunk,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    const limit = parseInt(options.limit);
    const topResults = results.slice(0, limit);

    console.log(dim(`Query embedding: ${queryTime.toFixed(1)}ms`));
    console.log(dim(`Searched ${results.length} chunks\n`));

    console.log(chalk.white.bold('Results:\n'));

    for (let i = 0; i < topResults.length; i++) {
      const r = topResults[i];
      const scoreColor = r.score > 0.7 ? chalk.green : r.score > 0.5 ? chalk.yellow : dim;

      console.log(`${i + 1}. ${scoreColor(`${(r.score * 100).toFixed(1)}%`)} - ${r.originalName}`);
      console.log(dim(`   Chunk ${r.chunkNumber}: ${r.text.slice(0, 120)}...`));
      console.log('');
    }
  });

/**
 * Benchmark Command
 */
program
  .command('benchmark')
  .description('Run indexing + search benchmarks')
  .option('--iterations <n>', 'Search iterations', '10')
  .action(async (options) => {
    console.log(chalk.blue('\n--- Benchmark ---\n'));

    // Check API
    const healthResp = await axios.get(`${API_URL}/health`);
    console.log(chalk.white.bold('Backend Info:'));
    console.log(dim(`  Provider: ${healthResp.data.provider}`));
    console.log(dim(`  Device: ${healthResp.data.device}`));
    console.log(dim(`  Model: ${healthResp.data.model}`));
    console.log('');

    const iterations = parseInt(options.iterations);
    const index = loadIndex();

    if (index.length === 0) {
      console.log(chalk.yellow('No documents indexed. Run "demo", "index", or "pipeline" first.'));
      process.exit(1);
    }

    const queries = [
      'Jan Wallace complaint',
      'breach of trust',
      'Supreme Court decision',
      'plaintiff damages',
      'legal precedent',
    ];

    const results: BenchmarkResult[] = [];

    console.log(chalk.white.bold('Running Query Benchmarks:'));
    console.log(dim(`  Iterations per query: ${iterations}`));
    console.log(dim(`  Test queries: ${queries.length}`));
    console.log('');

    for (const query of queries) {
      const times: number[] = [];

      console.log(chalk.white(`  "${query}"`));

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await embedQuery(query);
        times.push(performance.now() - start);
      }

      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const min = Math.min(...times);
      const max = Math.max(...times);

      results.push({
        operation: `Query: ${query}`,
        count: iterations,
        totalMs: times.reduce((a, b) => a + b, 0),
        avgMs: avg,
        minMs: min,
        maxMs: max,
      });

      console.log(chalk.green(`    Avg: ${avg.toFixed(1)}ms (min: ${min.toFixed(1)}ms, max: ${max.toFixed(1)}ms)`));
    }

    // Similarity search benchmark
    console.log(chalk.white('\n  Similarity Search (in-memory):'));
    const searchTimes: number[] = [];
    const queryEmbedding = await embedQuery(queries[0]);
    const totalChunks = index.reduce((sum, d) => sum + d.chunks.length, 0);

    for (let i = 0; i < iterations * 10; i++) {
      const start = performance.now();
      for (const doc of index) {
        for (const chunk of doc.chunks) {
          cosineSimilarity(queryEmbedding, chunk.embedding);
        }
      }
      searchTimes.push(performance.now() - start);
    }

    const searchAvg = searchTimes.reduce((a, b) => a + b, 0) / searchTimes.length;
    console.log(chalk.green(`    ${totalChunks} chunks: ${searchAvg.toFixed(2)}ms avg`));
    console.log(dim(`    Throughput: ${(totalChunks / searchAvg * 1000).toFixed(0)} comparisons/sec`));

    // Summary
    console.log(chalk.blue('\n--- Summary ---\n'));

    const allQueryTimes = results.filter(r => r.operation.startsWith('Query'));
    const overallAvg = allQueryTimes.reduce((sum, r) => sum + r.avgMs, 0) / allQueryTimes.length;

    console.log(chalk.white.bold('Query Embedding:'));
    console.log(dim(`  Average: ${overallAvg.toFixed(1)}ms`));
    console.log(dim(`  Throughput: ${(1000 / overallAvg).toFixed(1)} queries/sec`));
    console.log('');

    console.log(chalk.white.bold('Index Stats:'));
    console.log(dim(`  Documents: ${index.length}`));
    console.log(dim(`  Total chunks: ${totalChunks}`));
    console.log(dim(`  Avg chunks/doc: ${(totalChunks / index.length).toFixed(1)}`));

    const ocrDocs = index.filter(d => d.source === 'ocr').length;
    const textDocs = index.filter(d => d.source === 'text').length;
    console.log(dim(`  From text: ${textDocs}, From OCR: ${ocrDocs}`));
    console.log('');

    // Save results
    const benchmarkPath = join(OUTPUT_DIR, `benchmark-${Date.now()}.json`);
    mkdirSync(OUTPUT_DIR, { recursive: true });
    writeFileSync(benchmarkPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      backend: healthResp.data,
      index: {
        documents: index.length,
        chunks: totalChunks,
        ocrDocs,
        textDocs,
      },
      results,
    }, null, 2));
    console.log(dim(`Results saved to: ${benchmarkPath}`));
    console.log('');
  });

program.parse();
