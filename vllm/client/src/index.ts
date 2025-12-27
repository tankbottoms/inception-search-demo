/**
 * vLLM Hydra Client
 *
 * Demo and benchmark client for vLLM Hydra cluster services:
 *   - vllm-freelaw-modernbert (Embeddings) - port 8001
 *   - vllm-hunyuanOCR (OCR) - port 8003
 *   - vllm-gpt-oss-20b (Inference) - port 8004
 *
 * Commands:
 *   health    - Check service health
 *   test      - Run service tests
 *   ocr       - Process PDFs/images with OCR
 *   demo      - Run embeddings demo with similarity search
 *   pipeline  - Full pipeline (PDF -> OCR -> Embed -> Search)
 *   benchmark - Run performance benchmarks
 */

import { Command } from 'commander';
import axios, { AxiosError } from 'axios';
import chalk from 'chalk';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { join, basename } from 'path';
import sharp from 'sharp';

// Service URLs from environment or defaults
const EMBEDDINGS_URL = process.env.EMBEDDINGS_URL || `http://localhost:${process.env.EMBEDDINGS_PORT || 8001}`;
const OCR_URL = process.env.OCR_URL || `http://localhost:${process.env.OCR_PORT || 8003}`;
const INFERENCE_URL = process.env.INFERENCE_URL || `http://localhost:${process.env.INFERENCE_PORT || 8004}`;

const FILES_DIR = process.env.FILES_DIR || join(import.meta.dir, '../../files');
const OUTPUT_DIR = process.env.OUTPUT_DIR || join(import.meta.dir, '../../output');
const IMAGES_DIR = join(OUTPUT_DIR, 'images');

// ============================================================
// Types
// ============================================================

interface ModelsResponse {
  object: string;
  data: Array<{
    id: string;
    object: string;
    created: number;
    owned_by: string;
  }>;
}

interface EmbeddingResponse {
  object: string;
  data: Array<{
    object: string;
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface ChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ServiceStatus {
  name: string;
  url: string;
  healthy: boolean;
  responseTime: number;
  model?: string;
  error?: string;
}

interface IndexedDocument {
  id: number;
  filename: string;
  text: string;
  embedding: number[];
  source: 'text' | 'ocr';
  pageCount?: number;
  extractedAt: string;
}

interface SearchResult {
  documentId: number;
  filename: string;
  score: number;
  preview: string;
}

interface BenchmarkResult {
  service: string;
  operation: string;
  count: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  throughput: string;
}

// PDF parsing - dynamic import
let pdfParse: typeof import('pdf-parse');

// ============================================================
// Utilities
// ============================================================

async function initPdfParse() {
  if (!pdfParse) {
    pdfParse = (await import('pdf-parse')).default;
  }
  return pdfParse;
}

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function cleanText(text: string): string {
  return text
    .replace(/\s+/g, ' ')  // Collapse whitespace
    .replace(/[^\x20-\x7E]/g, '') // Remove non-ASCII
    .trim();
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

// ============================================================
// Model ID Fetching
// ============================================================

async function getModelId(serviceUrl: string): Promise<string | null> {
  try {
    const response = await axios.get<ModelsResponse>(`${serviceUrl}/v1/models`, { timeout: 5000 });
    if (response.data.data && response.data.data.length > 0) {
      return response.data.data[0].id;
    }
  } catch {
    // Service not available
  }
  return null;
}

// ============================================================
// Service Clients
// ============================================================

async function checkServiceHealth(url: string, name: string): Promise<ServiceStatus> {
  const startTime = performance.now();
  try {
    await axios.get(`${url}/health`, { timeout: 10000 });
    const modelId = await getModelId(url);
    return {
      name,
      url,
      healthy: true,
      responseTime: performance.now() - startTime,
      model: modelId || undefined,
    };
  } catch (error) {
    const errMsg = error instanceof AxiosError ? error.message : String(error);
    return {
      name,
      url,
      healthy: false,
      responseTime: performance.now() - startTime,
      error: errMsg,
    };
  }
}

async function getEmbeddings(text: string): Promise<number[]> {
  const modelId = await getModelId(EMBEDDINGS_URL);
  if (!modelId) throw new Error('Embeddings model not available');

  const response = await axios.post<EmbeddingResponse>(
    `${EMBEDDINGS_URL}/v1/embeddings`,
    {
      model: modelId,
      input: text,
    },
    { timeout: 60000 }
  );
  return response.data.data[0].embedding;
}

async function runOCR(imageBase64: string): Promise<string> {
  const modelId = await getModelId(OCR_URL);
  if (!modelId) throw new Error('OCR model not available');

  const response = await axios.post<ChatResponse>(
    `${OCR_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Extract all text from this image. Return only the extracted text, no explanations.' },
            { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
          ],
        },
      ],
      max_tokens: 4096,
    },
    { timeout: 300000 } // 5 min timeout for OCR
  );
  return response.data.choices[0].message.content;
}

async function runInference(prompt: string, systemPrompt?: string): Promise<string> {
  const modelId = await getModelId(INFERENCE_URL);
  if (!modelId) throw new Error('Inference model not available');

  const messages: Array<{ role: string; content: string }> = [];
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  const response = await axios.post<ChatResponse>(
    `${INFERENCE_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages,
      max_tokens: 2048,
      temperature: 0.7,
    },
    { timeout: 120000 }
  );
  return response.data.choices[0].message.content;
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

async function pdfToImages(pdfPath: string, outputDir: string): Promise<string[]> {
  const { fromPath } = await import('pdf2pic');

  const filename = basename(pdfPath, '.pdf');
  const imageDir = join(outputDir, filename);
  mkdirSync(imageDir, { recursive: true });

  const options = {
    density: 150,
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
    } catch {
      console.log(chalk.yellow(`    Warning: Could not convert page ${i}`));
    }
  }

  return imagePaths;
}

async function imageToBase64(imagePath: string): Promise<string> {
  const buffer = await sharp(imagePath)
    .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer();
  return buffer.toString('base64');
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
  .name('vllm-hydra-client')
  .description('Demo and benchmark client for vLLM Hydra cluster')
  .version('2.0.0');

/**
 * Health Command
 */
program
  .command('health')
  .description('Check health of all vLLM Hydra services')
  .action(async () => {
    console.log(chalk.cyan.bold('\n--- vLLM Hydra Health Check ---\n'));

    const services = [
      { name: 'Embeddings (vllm-freelaw-modernbert)', url: EMBEDDINGS_URL },
      { name: 'OCR (vllm-hunyuanOCR)', url: OCR_URL },
      { name: 'Inference (vllm-gpt-oss-20b)', url: INFERENCE_URL },
    ];

    let allHealthy = true;

    for (const service of services) {
      const status = await checkServiceHealth(service.url, service.name);

      if (status.healthy) {
        console.log(chalk.green.bold(`OK ${status.name}`));
        console.log(chalk.white(`   URL: ${status.url}`));
        console.log(chalk.white(`   Response: ${status.responseTime.toFixed(0)}ms`));
        if (status.model) {
          console.log(chalk.white(`   Model: ${status.model}`));
        }
      } else {
        console.log(chalk.red.bold(`X  ${status.name}`));
        console.log(chalk.white(`   URL: ${status.url}`));
        console.log(chalk.red(`   Error: ${status.error}`));
        allHealthy = false;
      }
      console.log('');
    }

    if (allHealthy) {
      console.log(chalk.green.bold('--- All services healthy ---\n'));
      process.exit(0);
    } else {
      console.log(chalk.yellow.bold('--- Some services unhealthy ---\n'));
      process.exit(1);
    }
  });

/**
 * Test Command
 */
program
  .command('test')
  .description('Run tests for all vLLM Hydra services')
  .action(async () => {
    console.log(chalk.cyan.bold('\n--- vLLM Hydra Service Tests ---\n'));

    const results: Array<{ test: string; passed: boolean; time: number; error?: string }> = [];

    // Test 1: Embeddings
    console.log(chalk.yellow.bold('Test 1: Embeddings Service'));
    const embStart = performance.now();
    try {
      const embedding = await getEmbeddings('This is a test sentence for embedding generation.');
      const embTime = performance.now() - embStart;

      if (embedding && embedding.length > 0) {
        console.log(chalk.green(`  OK Generated embedding with ${embedding.length} dimensions`));
        console.log(chalk.white(`  Time: ${embTime.toFixed(0)}ms`));
        results.push({ test: 'Embeddings', passed: true, time: embTime });
      } else {
        throw new Error('Empty embedding returned');
      }
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.log(chalk.red(`  X  Failed: ${errMsg}`));
      results.push({ test: 'Embeddings', passed: false, time: performance.now() - embStart, error: errMsg });
    }
    console.log('');

    // Test 2: OCR
    console.log(chalk.yellow.bold('Test 2: OCR Service (HunyuanOCR)'));
    const ocrStart = performance.now();
    try {
      // Create a simple test image with text using sharp
      const testImage = await sharp({
        create: {
          width: 200,
          height: 100,
          channels: 3,
          background: { r: 255, g: 255, b: 255 }
        }
      })
        .png()
        .toBuffer();

      const testImageBase64 = testImage.toString('base64');
      const ocrResult = await runOCR(testImageBase64);
      const ocrTime = performance.now() - ocrStart;

      console.log(chalk.green(`  OK OCR service responded`));
      console.log(chalk.white(`  Time: ${ocrTime.toFixed(0)}ms`));
      console.log(chalk.white(`  Response length: ${ocrResult.length} chars`));
      results.push({ test: 'OCR', passed: true, time: ocrTime });
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.log(chalk.red(`  X  Failed: ${errMsg}`));
      results.push({ test: 'OCR', passed: false, time: performance.now() - ocrStart, error: errMsg });
    }
    console.log('');

    // Test 3: Inference
    console.log(chalk.yellow.bold('Test 3: Inference Service (GPT-OSS)'));
    const infStart = performance.now();
    try {
      const response = await runInference('Say "Hello, World!" and nothing else.');
      const infTime = performance.now() - infStart;

      if (response && response.length > 0) {
        console.log(chalk.green(`  OK Inference completed`));
        console.log(chalk.white(`  Response: ${response.slice(0, 50)}...`));
        console.log(chalk.white(`  Time: ${infTime.toFixed(0)}ms`));
        results.push({ test: 'Inference', passed: true, time: infTime });
      } else {
        throw new Error('Empty response');
      }
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.log(chalk.red(`  X  Failed: ${errMsg}`));
      results.push({ test: 'Inference', passed: false, time: performance.now() - infStart, error: errMsg });
    }
    console.log('');

    // Summary
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;

    console.log(chalk.cyan.bold('--- Test Summary ---'));
    console.log(chalk.green(`  Passed: ${passed}`));
    if (failed > 0) {
      console.log(chalk.red(`  Failed: ${failed}`));
    }
    console.log('');

    process.exit(failed > 0 ? 1 : 0);
  });

/**
 * OCR Command
 */
program
  .command('ocr')
  .description('Process PDF or image with OCR')
  .option('--pdf <path>', 'PDF file to process')
  .option('--image <path>', 'Image file to process')
  .option('--output <path>', 'Output file for extracted text')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n--- OCR Processing ---\n'));

    // Check OCR service
    const ocrStatus = await checkServiceHealth(OCR_URL, 'OCR');
    if (!ocrStatus.healthy) {
      console.log(chalk.red(`X OCR service not available: ${ocrStatus.error}`));
      process.exit(1);
    }
    console.log(chalk.green(`OK OCR service ready (${ocrStatus.model})\n`));

    let allText = '';
    const startTime = performance.now();

    if (options.image) {
      // Process single image
      if (!existsSync(options.image)) {
        console.log(chalk.red(`X Image not found: ${options.image}`));
        process.exit(1);
      }

      console.log(chalk.white(`Processing image: ${options.image}`));
      const imageBase64 = await imageToBase64(options.image);
      allText = await runOCR(imageBase64);

    } else if (options.pdf) {
      // Process PDF
      if (!existsSync(options.pdf)) {
        console.log(chalk.red(`X PDF not found: ${options.pdf}`));
        process.exit(1);
      }

      console.log(chalk.white(`Processing PDF: ${options.pdf}`));
      console.log(chalk.white('  Converting to images...'));

      mkdirSync(IMAGES_DIR, { recursive: true });

      try {
        const imagePaths = await pdfToImages(options.pdf, IMAGES_DIR);
        if (imagePaths.length === 0) {
          console.log(chalk.red('X No pages could be converted'));
          process.exit(1);
        }

        console.log(chalk.green(`  OK Converted ${imagePaths.length} pages`));

        const pageTexts: string[] = [];
        for (let i = 0; i < imagePaths.length; i++) {
          console.log(chalk.white(`  OCR page ${i + 1}/${imagePaths.length}...`));
          const imageBase64 = await imageToBase64(imagePaths[i]);
          const text = await runOCR(imageBase64);
          pageTexts.push(text);

          // Clean up
          try { unlinkSync(imagePaths[i]); } catch {}
        }

        allText = pageTexts.join('\n\n--- Page Break ---\n\n');

      } catch (error) {
        console.log(chalk.red(`X PDF conversion failed: ${error}`));
        console.log(chalk.white('  Make sure graphicsmagick or imagemagick is installed'));
        process.exit(1);
      }

    } else {
      console.log(chalk.yellow('Provide --pdf or --image option'));
      process.exit(1);
    }

    const elapsed = performance.now() - startTime;

    console.log(chalk.green(`\nOK OCR Complete in ${formatTime(elapsed)}`));
    console.log(chalk.white(`Characters extracted: ${allText.length}`));

    console.log(chalk.white.bold('\nExtracted Text:'));
    console.log(chalk.white('-'.repeat(60)));
    console.log(allText.slice(0, 2000));
    if (allText.length > 2000) {
      console.log(chalk.white(`\n... (${allText.length - 2000} more characters)`));
    }
    console.log(chalk.white('-'.repeat(60)));

    if (options.output) {
      writeFileSync(options.output, allText);
      console.log(chalk.white(`\nSaved to: ${options.output}`));
    }
  });

/**
 * Demo Command - Embeddings with similarity search
 */
program
  .command('demo')
  .description('Run embeddings demo with similarity search')
  .option('--query <text>', 'Search query', 'legal document search')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n--- vLLM Hydra Embeddings Demo ---\n'));

    // Check embeddings service
    const embStatus = await checkServiceHealth(EMBEDDINGS_URL, 'Embeddings');
    if (!embStatus.healthy) {
      console.log(chalk.red(`X Embeddings service not available: ${embStatus.error}`));
      process.exit(1);
    }
    console.log(chalk.green(`OK Embeddings service ready (${embStatus.model})\n`));

    // Demo texts
    const documents = [
      'The plaintiff filed a motion for summary judgment in the civil case.',
      'Contract law requires consideration from both parties to be enforceable.',
      'The defendant argued that the statute of limitations had expired.',
      'Intellectual property rights include patents, trademarks, and copyrights.',
      'The court ruled in favor of the appellant and reversed the lower court decision.',
    ];

    console.log(chalk.white.bold('Generating embeddings for sample documents...\n'));

    const embeddings: Array<{ text: string; embedding: number[] }> = [];
    const totalStart = performance.now();

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      console.log(chalk.white(`${i + 1}. ${doc.slice(0, 60)}...`));

      const start = performance.now();
      const embedding = await getEmbeddings(doc);
      const elapsed = performance.now() - start;

      embeddings.push({ text: doc, embedding });
      console.log(chalk.green(`   OK ${embedding.length} dims in ${elapsed.toFixed(0)}ms`));
    }

    const totalTime = performance.now() - totalStart;
    console.log('');
    console.log(chalk.white(`Total embedding time: ${formatTime(totalTime)}`));
    console.log('');

    // Search
    console.log(chalk.cyan.bold('--- Similarity Search ---\n'));
    console.log(chalk.white(`Query: "${options.query}"\n`));

    const queryStart = performance.now();
    const queryEmbedding = await getEmbeddings(options.query);
    const queryTime = performance.now() - queryStart;
    console.log(chalk.white(`Query embedding: ${queryTime.toFixed(0)}ms\n`));

    // Calculate similarities
    const results = embeddings.map((doc, idx) => ({
      index: idx + 1,
      text: doc.text,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
    }));

    results.sort((a, b) => b.score - a.score);

    console.log(chalk.white.bold('Results (by similarity):\n'));
    for (const result of results) {
      const scoreColor = result.score > 0.7 ? chalk.green : result.score > 0.5 ? chalk.yellow : chalk.magenta;
      console.log(`${result.index}. ${scoreColor(`${(result.score * 100).toFixed(1)}%`)} - ${result.text}`);
    }

    console.log('');
    console.log(chalk.cyan.bold('--- Demo Complete ---\n'));
  });

/**
 * Pipeline Command - Full OCR to embedding pipeline
 */
program
  .command('pipeline')
  .description('Full pipeline: PDF -> OCR -> Embed -> Search')
  .option('--pdf-count <n>', 'Number of PDFs to process', '3')
  .option('--query <text>', 'Search query after indexing', 'legal complaint')
  .option('--force-ocr', 'Use OCR even if text extraction works', false)
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n--- Full OCR Pipeline ---\n'));

    // Check services
    const embStatus = await checkServiceHealth(EMBEDDINGS_URL, 'Embeddings');
    const ocrStatus = await checkServiceHealth(OCR_URL, 'OCR');

    if (!embStatus.healthy) {
      console.log(chalk.red(`X Embeddings service not available`));
      process.exit(1);
    }
    console.log(chalk.green(`OK Embeddings: ${embStatus.model}`));

    if (ocrStatus.healthy) {
      console.log(chalk.green(`OK OCR: ${ocrStatus.model}`));
    } else {
      console.log(chalk.yellow(`-- OCR: not available (will use text extraction only)`));
    }
    console.log('');

    // Get PDF files
    if (!existsSync(FILES_DIR)) {
      console.log(chalk.red(`X Files directory not found: ${FILES_DIR}`));
      console.log(chalk.white('  Create the directory and add PDF files'));
      process.exit(1);
    }

    let pdfFiles = readdirSync(FILES_DIR)
      .filter(f => f.endsWith('.pdf'))
      .slice(0, parseInt(options.pdfCount));

    if (pdfFiles.length === 0) {
      console.log(chalk.yellow('No PDF files found in files/ directory'));
      console.log(chalk.white(`  Add PDFs to: ${FILES_DIR}`));
      process.exit(1);
    }

    console.log(chalk.white.bold(`Processing ${pdfFiles.length} PDFs:\n`));

    mkdirSync(IMAGES_DIR, { recursive: true });

    const documents: IndexedDocument[] = [];
    let totalTime = 0;

    for (let i = 0; i < pdfFiles.length; i++) {
      const filename = pdfFiles[i];
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
            console.log(chalk.white('   Text extraction insufficient, trying OCR'));
            useOCR = true;
          } else {
            console.log(chalk.white(`   Text extracted: ${text.length} chars from ${pageCount} pages`));
          }
        } catch {
          console.log(chalk.white('   Text extraction failed, trying OCR'));
          useOCR = true;
        }
      }

      // Use OCR if needed
      if (useOCR && ocrStatus.healthy) {
        console.log(chalk.white('   Converting PDF to images...'));

        try {
          const imagePaths = await pdfToImages(pdfPath, IMAGES_DIR);
          pageCount = imagePaths.length;

          if (imagePaths.length === 0) {
            console.log(chalk.yellow('   Warning: No pages converted, skipping'));
            continue;
          }

          console.log(chalk.white(`   Running OCR on ${imagePaths.length} pages...`));

          const pageTexts: string[] = [];
          for (let p = 0; p < imagePaths.length; p++) {
            const imageBase64 = await imageToBase64(imagePaths[p]);
            const pageText = await runOCR(imageBase64);
            pageTexts.push(pageText);

            // Clean up
            try { unlinkSync(imagePaths[p]); } catch {}
          }

          text = pageTexts.join('\n\n--- Page Break ---\n\n');
          console.log(chalk.white(`   OCR complete: ${text.length} chars`));

        } catch (error) {
          console.log(chalk.yellow(`   Warning: OCR failed: ${error}`));
          continue;
        }
      } else if (useOCR && !ocrStatus.healthy) {
        console.log(chalk.yellow('   Warning: OCR needed but not available, skipping'));
        continue;
      }

      // Generate embeddings
      if (text.trim().length < 50) {
        console.log(chalk.yellow('   Warning: Insufficient text, skipping'));
        continue;
      }

      console.log(chalk.white('   Generating embeddings...'));

      try {
        const cleanedText = cleanText(text).slice(0, 1500); // Clean and limit text for 512-token model
        const embedding = await getEmbeddings(cleanedText);
        const elapsed = performance.now() - startTime;
        totalTime += elapsed;

        documents.push({
          id: i + 1,
          filename,
          text: cleanedText.slice(0, 500), // Store preview
          embedding,
          source: useOCR ? 'ocr' : 'text',
          pageCount,
          extractedAt: new Date().toISOString(),
        });

        console.log(chalk.green(`   OK ${embedding.length} dims in ${formatTime(elapsed)}`));

      } catch (error) {
        const errMsg = error instanceof Error ? error.message : String(error);
        console.log(chalk.red(`   X Embedding failed: ${errMsg}`));
      }

      console.log('');
    }

    // Save index
    saveIndex(documents);

    const ocrDocs = documents.filter(d => d.source === 'ocr').length;
    const textDocs = documents.filter(d => d.source === 'text').length;

    console.log(chalk.cyan.bold('--- Pipeline Complete ---\n'));
    console.log(chalk.white(`Documents indexed: ${documents.length}`));
    console.log(chalk.white(`  - From text extraction: ${textDocs}`));
    console.log(chalk.white(`  - From OCR: ${ocrDocs}`));
    console.log(chalk.white(`Total time: ${formatTime(totalTime)}`));
    console.log(chalk.white(`Index saved to: ${OUTPUT_DIR}/index.json`));
    console.log('');

    // Run search
    if (documents.length > 0) {
      console.log(chalk.cyan.bold('--- Search Results ---\n'));
      console.log(chalk.white(`Query: "${options.query}"\n`));

      const queryEmbedding = await getEmbeddings(options.query);

      const results: SearchResult[] = documents.map(doc => ({
        documentId: doc.id,
        filename: doc.filename,
        score: cosineSimilarity(queryEmbedding, doc.embedding),
        preview: doc.text.slice(0, 150),
      }));

      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, 5);

      for (let i = 0; i < topResults.length; i++) {
        const r = topResults[i];
        const scoreColor = r.score > 0.7 ? chalk.green : r.score > 0.5 ? chalk.yellow : chalk.magenta;
        console.log(`${i + 1}. ${scoreColor(`${(r.score * 100).toFixed(1)}%`)} - ${r.filename}`);
        console.log(chalk.white(`   ${r.preview}...`));
        console.log('');
      }
    }
  });

/**
 * Benchmark Command - Enhanced with bandwidth testing
 */
program
  .command('benchmark')
  .description('Run performance benchmarks with bandwidth testing')
  .option('--iterations <n>', 'Number of iterations per test', '10')
  .option('--concurrency <levels>', 'Concurrency levels to test (comma-separated)', '1,5,10,20')
  .option('--requests <n>', 'Total requests for throughput test', '50')
  .option('--skip-ocr', 'Skip OCR benchmark (slow)')
  .option('--skip-lb', 'Skip load balancer comparison')
  .action(async (options) => {
    console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
    console.log(chalk.cyan.bold('              vLLM Hydra Bandwidth Benchmark'));
    console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

    const iterations = parseInt(options.iterations);
    const totalRequests = parseInt(options.requests);
    const concurrencyLevels = options.concurrency.split(',').map((n: string) => parseInt(n.trim()));
    const results: BenchmarkResult[] = [];

    // Endpoint configurations
    const SINGLE_ENDPOINT = EMBEDDINGS_URL;  // Port 8001
    const LB_ENDPOINT = process.env.LB_URL || 'http://localhost:8000';  // Load balanced

    // Check services
    console.log(chalk.white.bold('Service Discovery'));
    console.log(chalk.dim('-'.repeat(50)));

    const singleStatus = await checkServiceHealth(SINGLE_ENDPOINT, 'Single Replica');
    const lbStatus = await checkServiceHealth(LB_ENDPOINT, 'Load Balanced');
    const ocrStatus = await checkServiceHealth(OCR_URL, 'OCR');
    const infStatus = await checkServiceHealth(INFERENCE_URL, 'Inference');

    console.log(singleStatus.healthy
      ? chalk.green(`  [OK] Single Replica (${SINGLE_ENDPOINT})`) + chalk.dim(` - ${singleStatus.model}`)
      : chalk.red(`  [X] Single Replica: ${singleStatus.error}`));
    console.log(lbStatus.healthy
      ? chalk.green(`  [OK] Load Balanced (${LB_ENDPOINT})`) + chalk.dim(` - Traefik LB`)
      : chalk.dim(`  [--] Load Balanced: not available`));
    console.log(ocrStatus.healthy
      ? chalk.green(`  [OK] OCR`) + chalk.dim(` - ${ocrStatus.model}`)
      : chalk.dim(`  [--] OCR: not available`));
    console.log(infStatus.healthy
      ? chalk.green(`  [OK] Inference`) + chalk.dim(` - ${infStatus.model}`)
      : chalk.dim(`  [--] Inference: not available`));
    console.log('');

    if (!singleStatus.healthy) {
      console.log(chalk.red('Embeddings service not available. Start with: ./scripts/hydra-start.sh'));
      process.exit(1);
    }

    // ================================================================
    // Latency Benchmark (Sequential)
    // ================================================================
    console.log(chalk.white.bold('Latency Benchmark (Sequential)'));
    console.log(chalk.dim('-'.repeat(50)));
    console.log(chalk.white(`  Iterations: ${iterations}`));
    console.log('');

    const testTexts = [
      { name: 'Short', text: 'Short text for testing.' },
      { name: 'Medium', text: 'A medium length text that contains more words and context for embedding generation benchmark.' },
      { name: 'Long', text: 'A longer document that simulates a real-world use case with multiple sentences. This text contains legal terminology and concepts that might be found in court documents. The plaintiff alleged breach of contract and sought compensatory damages.' },
    ];

    // Get model ID once
    const modelId = await getModelId(SINGLE_ENDPOINT);
    if (!modelId) {
      console.log(chalk.red('Could not get model ID'));
      process.exit(1);
    }

    for (const { name, text } of testTexts) {
      const times: number[] = [];
      process.stdout.write(chalk.white(`  ${name} (${text.length} chars): `));

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        try {
          await axios.post(`${SINGLE_ENDPOINT}/v1/embeddings`, {
            model: modelId,
            input: text,
          }, { timeout: 30000 });
          times.push(performance.now() - start);
        } catch {
          // Skip failed
        }
      }

      if (times.length > 0) {
        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        const p95 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)] || max;

        console.log(chalk.green.bold(`${avg.toFixed(1)}ms`) +
          chalk.dim(` (min: ${min.toFixed(1)}, max: ${max.toFixed(1)}, p95: ${p95.toFixed(1)})`));

        results.push({
          service: 'Embeddings',
          operation: `Latency - ${name}`,
          count: times.length,
          avgMs: avg,
          minMs: min,
          maxMs: max,
          throughput: `${(1000 / avg).toFixed(1)} req/s`,
        });
      } else {
        console.log(chalk.red.bold('failed'));
      }
    }
    console.log('');

    // ================================================================
    // Throughput Benchmark (Concurrent)
    // ================================================================
    console.log(chalk.white.bold('Throughput Benchmark (Concurrent)'));
    console.log(chalk.dim('-'.repeat(50)));

    const throughputText = 'Document for throughput testing with realistic content.';

    // Helper function for concurrent requests
    async function runConcurrentBenchmark(
      endpoint: string,
      concurrency: number,
      total: number,
      label: string
    ): Promise<{ totalMs: number; successCount: number; avgMs: number; rps: number }> {
      const startTime = performance.now();
      let successCount = 0;
      const times: number[] = [];

      // Create batches
      const batches = Math.ceil(total / concurrency);

      for (let batch = 0; batch < batches; batch++) {
        const batchSize = Math.min(concurrency, total - batch * concurrency);
        const promises = [];

        for (let i = 0; i < batchSize; i++) {
          const reqStart = performance.now();
          promises.push(
            axios.post(`${endpoint}/v1/embeddings`, {
              model: modelId,
              input: `${throughputText} Request ${batch * concurrency + i + 1}.`,
            }, { timeout: 60000 })
              .then(() => {
                times.push(performance.now() - reqStart);
                successCount++;
              })
              .catch(() => {})
          );
        }

        await Promise.all(promises);
      }

      const totalMs = performance.now() - startTime;
      const avgMs = times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
      const rps = (successCount / totalMs) * 1000;

      return { totalMs, successCount, avgMs, rps };
    }

    // Test single endpoint at different concurrency levels
    console.log(chalk.white('\n  Single Replica Performance:'));

    for (const concurrency of concurrencyLevels) {
      process.stdout.write(chalk.white(`    ${totalRequests} reqs @ ${concurrency} concurrent: `));

      const result = await runConcurrentBenchmark(SINGLE_ENDPOINT, concurrency, totalRequests, 'Single');

      console.log(chalk.green.bold(`${result.rps.toFixed(1)} req/s`) +
        chalk.dim(` (${result.totalMs.toFixed(0)}ms total, ${result.successCount}/${totalRequests} success)`));

      results.push({
        service: 'Single Endpoint',
        operation: `Throughput @ ${concurrency} concurrent`,
        count: result.successCount,
        avgMs: result.avgMs,
        minMs: result.avgMs * 0.8,  // Estimate
        maxMs: result.avgMs * 1.5,  // Estimate
        throughput: `${result.rps.toFixed(1)} req/s`,
      });
    }

    // Test load-balanced endpoint if available
    if (lbStatus.healthy && !options.skipLb) {
      console.log(chalk.white('\n  Load Balanced Performance:'));

      for (const concurrency of concurrencyLevels) {
        process.stdout.write(chalk.white(`    ${totalRequests} reqs @ ${concurrency} concurrent: `));

        const result = await runConcurrentBenchmark(LB_ENDPOINT, concurrency, totalRequests, 'LB');

        console.log(chalk.green.bold(`${result.rps.toFixed(1)} req/s`) +
          chalk.dim(` (${result.totalMs.toFixed(0)}ms total, ${result.successCount}/${totalRequests} success)`));

        results.push({
          service: 'Load Balanced',
          operation: `Throughput @ ${concurrency} concurrent`,
          count: result.successCount,
          avgMs: result.avgMs,
          minMs: result.avgMs * 0.8,
          maxMs: result.avgMs * 1.5,
          throughput: `${result.rps.toFixed(1)} req/s`,
        });
      }

      // Direct comparison at highest concurrency
      const maxConcurrency = Math.max(...concurrencyLevels);
      console.log(chalk.white('\n  Head-to-Head Comparison:'));

      const singleResult = await runConcurrentBenchmark(SINGLE_ENDPOINT, maxConcurrency, totalRequests, 'Single');
      const lbResult = await runConcurrentBenchmark(LB_ENDPOINT, maxConcurrency, totalRequests, 'LB');

      const improvement = ((lbResult.rps / singleResult.rps) - 1) * 100;
      const improvementColor = improvement > 0 ? chalk.green.bold : chalk.red.bold;

      console.log(chalk.white(`    Single Replica: `) + chalk.yellow.bold(`${singleResult.rps.toFixed(1)} req/s`));
      console.log(chalk.white(`    Load Balanced:  `) + chalk.green.bold(`${lbResult.rps.toFixed(1)} req/s`));
      console.log(chalk.white(`    Improvement:    `) + improvementColor(`${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%`));
    }
    console.log('');

    // ================================================================
    // Batch Embedding Test
    // ================================================================
    console.log(chalk.white.bold('Batch Embedding Test'));
    console.log(chalk.dim('-'.repeat(50)));

    const batchSizes = [1, 5, 10, 20];
    const batchTexts = Array.from({ length: 20 }, (_, i) =>
      `Document ${i + 1}: Legal content about contracts, agreements, and court proceedings.`
    );

    for (const batchSize of batchSizes) {
      process.stdout.write(chalk.white(`  Batch of ${batchSize}: `));

      const start = performance.now();
      try {
        await axios.post(`${SINGLE_ENDPOINT}/v1/embeddings`, {
          model: modelId,
          input: batchTexts.slice(0, batchSize),
        }, { timeout: 60000 });

        const elapsed = performance.now() - start;
        const perDoc = elapsed / batchSize;

        console.log(chalk.green.bold(`${elapsed.toFixed(0)}ms total`) +
          chalk.dim(` (${perDoc.toFixed(1)}ms/doc, ${(1000 / perDoc).toFixed(1)} docs/s)`));

        results.push({
          service: 'Embeddings',
          operation: `Batch (${batchSize} docs)`,
          count: batchSize,
          avgMs: perDoc,
          minMs: perDoc,
          maxMs: perDoc,
          throughput: `${(1000 / perDoc).toFixed(1)} docs/s`,
        });
      } catch (error) {
        console.log(chalk.red.bold(`failed: ${error}`));
      }
    }
    console.log('');

    // ================================================================
    // OCR Benchmark (if available and not skipped)
    // ================================================================
    if (ocrStatus.healthy && !options.skipOcr) {
      console.log(chalk.white.bold('OCR Benchmark'));
      console.log(chalk.dim('-'.repeat(50)));

      const testImage = await sharp({
        create: { width: 400, height: 200, channels: 3, background: { r: 255, g: 255, b: 255 } }
      }).png().toBuffer();

      const testImageBase64 = testImage.toString('base64');

      process.stdout.write(chalk.white(`  Simple image (400x200): `));

      const start = performance.now();
      try {
        await runOCR(testImageBase64);
        const elapsed = performance.now() - start;
        console.log(chalk.green.bold(`${elapsed.toFixed(0)}ms`));

        results.push({
          service: 'OCR',
          operation: 'Simple image',
          count: 1,
          avgMs: elapsed,
          minMs: elapsed,
          maxMs: elapsed,
          throughput: `${(1000 / elapsed).toFixed(2)} req/s`,
        });
      } catch (error) {
        console.log(chalk.red.bold(`failed: ${error}`));
      }
      console.log('');
    } else if (options.skipOcr) {
      console.log(chalk.dim('OCR Benchmark: Skipped (--skip-ocr)\n'));
    }

    // ================================================================
    // Summary Report
    // ================================================================
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log(chalk.cyan.bold('                      Benchmark Summary'));
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log('');

    // Group results by service
    const byService = results.reduce((acc, r) => {
      if (!acc[r.service]) acc[r.service] = [];
      acc[r.service].push(r);
      return acc;
    }, {} as Record<string, BenchmarkResult[]>);

    for (const [service, serviceResults] of Object.entries(byService)) {
      console.log(chalk.white.bold(`${service}:`));
      for (const r of serviceResults) {
        console.log(chalk.white(`  ${r.operation.padEnd(30)} `) + chalk.green.bold(`${r.throughput.padStart(12)}`));
      }
      console.log('');
    }

    // Best throughput
    const throughputResults = results.filter(r => r.operation.includes('Throughput'));
    if (throughputResults.length > 0) {
      const best = throughputResults.reduce((a, b) =>
        parseFloat(a.throughput) > parseFloat(b.throughput) ? a : b
      );
      console.log(chalk.white.bold('Peak Performance:'));
      console.log(chalk.green.bold(`  ${best.service} - ${best.operation}: ${best.throughput}`));
      console.log('');
    }

    // Save rich JSON report
    const report = {
      timestamp: new Date().toISOString(),
      config: {
        iterations,
        totalRequests,
        concurrencyLevels,
        singleEndpoint: SINGLE_ENDPOINT,
        lbEndpoint: LB_ENDPOINT,
      },
      services: {
        singleReplica: {
          url: SINGLE_ENDPOINT,
          available: singleStatus.healthy,
          model: singleStatus.model,
          responseTime: singleStatus.responseTime,
        },
        loadBalanced: {
          url: LB_ENDPOINT,
          available: lbStatus.healthy,
          responseTime: lbStatus.responseTime,
        },
        ocr: {
          url: OCR_URL,
          available: ocrStatus.healthy,
          model: ocrStatus.model,
        },
        inference: {
          url: INFERENCE_URL,
          available: infStatus.healthy,
          model: infStatus.model,
        },
      },
      results,
      summary: {
        totalTests: results.length,
        avgLatencyMs: results.filter(r => r.operation.includes('Latency'))
          .reduce((sum, r) => sum + r.avgMs, 0) / results.filter(r => r.operation.includes('Latency')).length || 0,
        peakThroughput: throughputResults.length > 0
          ? Math.max(...throughputResults.map(r => parseFloat(r.throughput)))
          : 0,
        lbImprovement: lbStatus.healthy ? (() => {
          const singleMax = throughputResults.filter(r => r.service === 'Single Endpoint')
            .reduce((max, r) => Math.max(max, parseFloat(r.throughput)), 0);
          const lbMax = throughputResults.filter(r => r.service === 'Load Balanced')
            .reduce((max, r) => Math.max(max, parseFloat(r.throughput)), 0);
          return singleMax > 0 ? ((lbMax / singleMax) - 1) * 100 : 0;
        })() : null,
      },
    };

    const benchmarkPath = join(OUTPUT_DIR, `benchmark-${Date.now()}.json`);
    mkdirSync(OUTPUT_DIR, { recursive: true });
    writeFileSync(benchmarkPath, JSON.stringify(report, null, 2));

    console.log(chalk.white(`Report saved: ${benchmarkPath}`));
    console.log('');
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log(chalk.green.bold('                    Benchmark Complete!'));
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log('');
  });

program.parse();
