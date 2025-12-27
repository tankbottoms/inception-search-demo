/**
 * vLLM Hydra - Comprehensive Demo
 *
 * Full workflow:
 *   1. Convert PDF pages to images
 *   2. OCR each page using HunyuanOCR
 *   3. Extract text directly from PDF
 *   4. Compare OCR vs extracted text
 *   5. Use GPT-OSS to generate best quality markdown
 *   6. Generate embeddings
 *   7. Perform semantic search
 *   8. Generate citation using GPT-OSS
 *   9. Collect benchmark stats
 *   10. Monitor GPU utilization
 */

import { Command } from 'commander';
import axios, { AxiosError } from 'axios';
import chalk from 'chalk';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { execSync } from 'child_process';
import { join, basename } from 'path';
import sharp from 'sharp';

// Service URLs
const EMBEDDINGS_URL = process.env.EMBEDDINGS_URL || `http://localhost:${process.env.EMBEDDINGS_PORT || 8001}`;
const OCR_URL = process.env.OCR_URL || `http://localhost:${process.env.OCR_PORT || 8003}`;
const INFERENCE_URL = process.env.INFERENCE_URL || `http://localhost:${process.env.INFERENCE_PORT || 8004}`;

const FILES_DIR = process.env.FILES_DIR || join(import.meta.dir, '../../files');
const OUTPUT_DIR = process.env.OUTPUT_DIR || join(import.meta.dir, '../../output');
const IMAGES_DIR = join(OUTPUT_DIR, 'images');

// PDF parsing
let pdfParse: typeof import('pdf-parse');

// ============================================================
// Types
// ============================================================

interface BenchmarkStats {
  operation: string;
  startTime: number;
  endTime: number;
  durationMs: number;
  gpuUtilBefore?: number;
  gpuUtilAfter?: number;
  gpuMemoryMB?: number;
  success: boolean;
  error?: string;
}

interface DemoResults {
  timestamp: string;
  pdfFile: string;
  pageCount: number;
  ocrText: string;
  extractedText: string;
  mergedMarkdown: string;
  embedding: number[];
  searchResults: Array<{ query: string; score: number }>;
  citation: string;
  benchmarks: BenchmarkStats[];
  totalDurationMs: number;
  gpuStats: {
    avgUtilization: number;
    maxUtilization: number;
    maxMemoryMB: number;
  };
}

// ============================================================
// GPU Monitoring
// ============================================================

interface GpuStats {
  utilization: number;
  memoryUsedMB: number;
  memoryTotalMB: number;
  temperature: number;
}

// Check nvidia-smi availability once at startup
let nvidiaSmiAvailable: boolean | null = null;

function checkNvidiaSmi(): boolean {
  if (nvidiaSmiAvailable !== null) return nvidiaSmiAvailable;
  try {
    execSync('which nvidia-smi', { encoding: 'utf-8', stdio: 'pipe' });
    nvidiaSmiAvailable = true;
  } catch {
    nvidiaSmiAvailable = false;
  }
  return nvidiaSmiAvailable;
}

function getGpuStats(): GpuStats | null {
  if (!checkNvidiaSmi()) return null;

  try {
    const output = execSync(
      'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits',
      { encoding: 'utf-8', timeout: 5000, stdio: 'pipe' }
    ).trim();

    const parts = output.split(',').map(s => s.trim());
    const parseVal = (s: string) => {
      const num = parseFloat(s);
      return isNaN(num) ? 0 : num;
    };

    return {
      utilization: parseVal(parts[0]),
      memoryUsedMB: parseVal(parts[1]),
      memoryTotalMB: parseVal(parts[2]) || 1,
      temperature: parseVal(parts[3]),
    };
  } catch {
    return null;
  }
}

function formatGpuStats(stats: GpuStats | null): string {
  if (!stats) return chalk.dim('GPU: (host only)');

  const utilColor = stats.utilization > 80 ? chalk.red.bold :
                    stats.utilization > 50 ? chalk.yellow.bold : chalk.green.bold;
  const memPercent = (stats.memoryUsedMB / stats.memoryTotalMB * 100).toFixed(0);

  return `${utilColor(`GPU: ${stats.utilization}%`)} | Mem: ${memPercent}% (${stats.memoryUsedMB.toFixed(0)}MB)`;
}

// ============================================================
// imgcat Support for iTerm2
// ============================================================

function outputImageWithImgcat(imagePath: string): void {
  try {
    execSync(`imgcat "${imagePath}"`, { stdio: 'inherit' });
  } catch {
    console.log(chalk.dim(`  [Image: ${imagePath}]`));
  }
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

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function cleanText(text: string): string {
  return text
    .replace(/\s+/g, ' ')
    .replace(/[^\x20-\x7E\n]/g, '')
    .trim();
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============================================================
// API Clients
// ============================================================

async function getModelId(url: string): Promise<string | null> {
  try {
    const response = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    return response.data.data?.[0]?.id || null;
  } catch {
    return null;
  }
}

async function checkService(url: string): Promise<boolean> {
  try {
    await axios.get(`${url}/health`, { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
}

async function getEmbeddings(text: string): Promise<number[]> {
  const modelId = await getModelId(EMBEDDINGS_URL);
  if (!modelId) throw new Error('Embeddings model not available');

  const response = await axios.post(
    `${EMBEDDINGS_URL}/v1/embeddings`,
    { model: modelId, input: text },
    { timeout: 60000 }
  );
  return response.data.data[0].embedding;
}

async function runOCR(imageBase64: string): Promise<string> {
  const modelId = await getModelId(OCR_URL);
  if (!modelId) throw new Error('OCR model not available');

  const response = await axios.post(
    `${OCR_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'Extract all text from this image. Preserve the formatting and structure. Return only the extracted text.' },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
        ],
      }],
      max_tokens: 4096,
    },
    { timeout: 300000 }
  );
  return response.data.choices[0].message.content;
}

async function runInference(prompt: string, systemPrompt?: string): Promise<string> {
  const modelId = await getModelId(INFERENCE_URL);
  if (!modelId) throw new Error('Inference model not available');

  const messages: Array<{ role: string; content: string }> = [];
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  messages.push({ role: 'user', content: prompt });

  const response = await axios.post(
    `${INFERENCE_URL}/v1/chat/completions`,
    { model: modelId, messages, max_tokens: 2048, temperature: 0.3 },
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
      if (result.path) imagePaths.push(result.path);
    } catch {
      // Skip failed pages
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
// Demo Command
// ============================================================

const program = new Command();

program
  .name('vllm-demo')
  .description('Comprehensive vLLM Hydra demo with OCR, embeddings, and citations')
  .version('1.0.0');

program
  .command('run')
  .description('Run the full demo workflow')
  .option('--pdf <path>', 'PDF file to process', '')
  .option('--pages <n>', 'Max pages to OCR (0=all)', '3')
  .option('--query <text>', 'Search query', 'legal complaint damages')
  .option('--output <path>', 'Output directory', OUTPUT_DIR)
  .option('--no-inference', 'Skip inference step')
  .option('--show-images', 'Display images using imgcat (iTerm2)')
  .action(async (options) => {
    const startTime = performance.now();
    const benchmarks: BenchmarkStats[] = [];

    console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
    console.log(chalk.cyan.bold('                    vLLM Hydra - Comprehensive Demo'));
    console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

    // Check GPU
    const gpuStats = getGpuStats();
    console.log(chalk.white.bold('GPU Status:'), formatGpuStats(gpuStats));
    console.log('');

    // Check services
    console.log(chalk.white.bold('Checking Services...\n'));

    const embAvailable = await checkService(EMBEDDINGS_URL);
    const ocrAvailable = await checkService(OCR_URL);
    const infAvailable = await checkService(INFERENCE_URL);

    console.log(embAvailable ? chalk.green.bold('  [OK] Embeddings') : chalk.red.bold('  [X] Embeddings'));
    console.log(ocrAvailable ? chalk.green.bold('  [OK] OCR (HunyuanOCR)') : chalk.red.bold('  [X] OCR'));
    console.log(infAvailable ? chalk.green.bold('  [OK] Inference') : chalk.dim('  [--] Inference (optional)'));
    console.log('');

    if (!embAvailable || !ocrAvailable) {
      console.log(chalk.red('Required services not available. Start with: ./scripts/hydra-start.sh'));
      process.exit(1);
    }

    // Find PDF
    let pdfPath = options.pdf;
    if (!pdfPath) {
      const pdfs = readdirSync(FILES_DIR).filter(f => f.endsWith('.pdf'));
      if (pdfs.length === 0) {
        console.log(chalk.red(`No PDF files found in ${FILES_DIR}`));
        process.exit(1);
      }
      pdfPath = join(FILES_DIR, pdfs[0]);
    }

    if (!existsSync(pdfPath)) {
      console.log(chalk.red(`PDF not found: ${pdfPath}`));
      process.exit(1);
    }

    const pdfName = basename(pdfPath);
    console.log(chalk.cyan.bold(`Processing: ${pdfName}`));
    console.log(chalk.gray(`Path: ${pdfPath}`));
    console.log('');

    mkdirSync(options.output, { recursive: true });
    mkdirSync(IMAGES_DIR, { recursive: true });

    // ================================================================
    // STEP 1: PDF to Images
    // ================================================================
    console.log(chalk.yellow.bold('Step 1: Converting PDF to Images'));
    console.log(chalk.gray('-'.repeat(50)));

    let gpuBefore = getGpuStats();
    let stepStart = performance.now();

    const imagePaths = await pdfToImages(pdfPath, IMAGES_DIR);
    const maxPages = options.pages === '0' ? imagePaths.length : Math.min(parseInt(options.pages), imagePaths.length);

    let stepEnd = performance.now();
    let gpuAfter = getGpuStats();

    benchmarks.push({
      operation: 'PDF to Images',
      startTime: stepStart,
      endTime: stepEnd,
      durationMs: stepEnd - stepStart,
      gpuUtilBefore: gpuBefore?.utilization,
      gpuUtilAfter: gpuAfter?.utilization,
      success: imagePaths.length > 0,
    });

    console.log(chalk.green(`  Converted ${imagePaths.length} pages in ${formatTime(stepEnd - stepStart)}`));
    console.log(chalk.gray(`  Processing first ${maxPages} pages`));
    console.log('');

    // ================================================================
    // STEP 2: OCR Each Page
    // ================================================================
    console.log(chalk.white.bold('Step 2: OCR Processing (HunyuanOCR)'));
    console.log(chalk.dim('-'.repeat(50)));

    const ocrTexts: string[] = [];
    let totalOcrTime = 0;

    for (let i = 0; i < maxPages; i++) {
      const pageNum = i + 1;
      console.log(chalk.white.bold(`\n  Page ${pageNum}/${maxPages}:`));

      // Show image if requested
      if (options.showImages && existsSync(imagePaths[i])) {
        console.log('');
        outputImageWithImgcat(imagePaths[i]);
        console.log('');
      }

      gpuBefore = getGpuStats();
      stepStart = performance.now();

      try {
        const imageBase64 = await imageToBase64(imagePaths[i]);
        const text = await runOCR(imageBase64);
        ocrTexts.push(text);

        stepEnd = performance.now();
        gpuAfter = getGpuStats();
        totalOcrTime += stepEnd - stepStart;

        const gpuInfo = gpuAfter ? chalk.cyan.bold(` [GPU: ${gpuAfter.utilization}%]`) : '';
        console.log(chalk.green.bold(`  ✓ ${text.length} chars in ${formatTime(stepEnd - stepStart)}${gpuInfo}`));

        // Show OCR'd text preview
        console.log(chalk.dim('  ' + '-'.repeat(40)));
        const textPreview = text.slice(0, 300).replace(/\n/g, '\n  ');
        console.log(chalk.white(`  ${textPreview}`));
        if (text.length > 300) console.log(chalk.dim(`  ... (${text.length - 300} more chars)`));
        console.log(chalk.dim('  ' + '-'.repeat(40)));

        benchmarks.push({
          operation: `OCR Page ${pageNum}`,
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          gpuUtilBefore: gpuBefore?.utilization,
          gpuUtilAfter: gpuAfter?.utilization,
          gpuMemoryMB: gpuAfter?.memoryUsedMB,
          success: true,
        });
      } catch (error) {
        stepEnd = performance.now();
        console.log(chalk.red.bold(`  ✗ Failed: ${error}`));
        benchmarks.push({
          operation: `OCR Page ${pageNum}`,
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          success: false,
          error: String(error),
        });
      }

      // Cleanup image (unless showing images)
      if (!options.showImages) {
        try { unlinkSync(imagePaths[i]); } catch {}
      }
    }

    const ocrText = ocrTexts.join('\n\n--- Page Break ---\n\n');
    console.log(chalk.green.bold(`\n  Total: ${ocrText.length} chars in ${formatTime(totalOcrTime)}`));
    console.log('');

    // ================================================================
    // STEP 3: Extract Text from PDF
    // ================================================================
    console.log(chalk.white.bold('Step 3: Direct PDF Text Extraction'));
    console.log(chalk.dim('-'.repeat(50)));

    stepStart = performance.now();
    let extractedText = '';

    try {
      extractedText = await extractPdfText(pdfPath);
      stepEnd = performance.now();

      benchmarks.push({
        operation: 'PDF Text Extraction',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        success: true,
      });

      console.log(chalk.green(`  Extracted ${extractedText.length} chars in ${formatTime(stepEnd - stepStart)}`));
    } catch (error) {
      stepEnd = performance.now();
      console.log(chalk.yellow(`  Warning: ${error}`));
      benchmarks.push({
        operation: 'PDF Text Extraction',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        success: false,
        error: String(error),
      });
    }
    console.log('');

    // ================================================================
    // STEP 4: Compare OCR vs Extracted Text
    // ================================================================
    console.log(chalk.white.bold('Step 4: Text Comparison'));
    console.log(chalk.dim('-'.repeat(50)));

    const ocrClean = cleanText(ocrText);
    const extractedClean = cleanText(extractedText);

    const ocrWords = ocrClean.split(/\s+/).length;
    const extractedWords = extractedClean.split(/\s+/).length;

    console.log(chalk.white(`  OCR: ${ocrWords} words | Extracted: ${extractedWords} words`));

    // Simple overlap calculation
    const ocrWordSet = new Set(ocrClean.toLowerCase().split(/\s+/));
    const extractedWordSet = new Set(extractedClean.toLowerCase().split(/\s+/));
    const intersection = [...ocrWordSet].filter(w => extractedWordSet.has(w));
    const overlap = intersection.length / Math.max(ocrWordSet.size, extractedWordSet.size) * 100;

    console.log(chalk.cyan.bold(`  Word overlap: ${overlap.toFixed(1)}%`));
    console.log('');

    // ================================================================
    // STEP 5: Generate Merged Markdown
    // ================================================================
    let mergedMarkdown = '';

    if (infAvailable && options.inference !== false) {
      console.log(chalk.white.bold('Step 5: Generate Best Quality Markdown (Inference)'));
      console.log(chalk.dim('-'.repeat(50)));

      gpuBefore = getGpuStats();
      stepStart = performance.now();

      try {
        const prompt = `You have two versions of text from the same document:

**OCR Version:**
${ocrClean.slice(0, 3000)}

**Extracted Version:**
${extractedClean.slice(0, 3000)}

Create a clean, well-formatted markdown document that combines the best parts of both versions. Fix any OCR errors using context from the extracted text. Use proper markdown formatting (headers, lists, etc).`;

        mergedMarkdown = await runInference(prompt, 'You are a document processing assistant. Create clean, accurate markdown from document text.');

        stepEnd = performance.now();
        gpuAfter = getGpuStats();

        console.log(chalk.green(`  Generated ${mergedMarkdown.length} chars in ${formatTime(stepEnd - stepStart)}`));
        console.log(chalk.cyan(`  ${formatGpuStats(gpuAfter)}`));

        benchmarks.push({
          operation: 'Markdown Generation',
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          gpuUtilBefore: gpuBefore?.utilization,
          gpuUtilAfter: gpuAfter?.utilization,
          gpuMemoryMB: gpuAfter?.memoryUsedMB,
          success: true,
        });
      } catch (error) {
        stepEnd = performance.now();
        console.log(chalk.yellow(`  Skipped: ${error}`));
        mergedMarkdown = ocrClean;  // Fallback to OCR

        benchmarks.push({
          operation: 'Markdown Generation',
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          success: false,
          error: String(error),
        });
      }
    } else {
      console.log(chalk.white.bold('Step 5: Markdown Generation'));
      console.log(chalk.dim('-'.repeat(50)));
      console.log(chalk.dim('  Skipped (Inference not available)'));
      mergedMarkdown = ocrClean;
    }
    console.log('');

    // Save markdown
    const mdPath = join(options.output, `${basename(pdfPath, '.pdf')}.md`);
    writeFileSync(mdPath, mergedMarkdown);
    console.log(chalk.white(`  Saved: ${mdPath}`));
    console.log('');

    // ================================================================
    // STEP 6: Generate Embeddings
    // ================================================================
    console.log(chalk.white.bold('Step 6: Generate Embeddings'));
    console.log(chalk.dim('-'.repeat(50)));

    gpuBefore = getGpuStats();
    stepStart = performance.now();

    let embedding: number[] = [];
    try {
      const textForEmbed = cleanText(mergedMarkdown || ocrClean).slice(0, 1500);
      embedding = await getEmbeddings(textForEmbed);

      stepEnd = performance.now();
      gpuAfter = getGpuStats();

      console.log(chalk.green(`  Generated ${embedding.length}-dim embedding in ${formatTime(stepEnd - stepStart)}`));
      console.log(chalk.cyan(`  ${formatGpuStats(gpuAfter)}`));

      benchmarks.push({
        operation: 'Embedding Generation',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        gpuUtilBefore: gpuBefore?.utilization,
        gpuUtilAfter: gpuAfter?.utilization,
        success: true,
      });
    } catch (error) {
      stepEnd = performance.now();
      console.log(chalk.red(`  Failed: ${error}`));
      benchmarks.push({
        operation: 'Embedding Generation',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        success: false,
        error: String(error),
      });
    }
    console.log('');

    // ================================================================
    // STEP 7: Semantic Search
    // ================================================================
    console.log(chalk.white.bold('Step 7: Semantic Search'));
    console.log(chalk.dim('-'.repeat(50)));
    console.log(chalk.white(`  Query: "${options.query}"`));

    const searchResults: Array<{ query: string; score: number }> = [];

    stepStart = performance.now();
    try {
      const queryEmbedding = await getEmbeddings(options.query);
      const similarity = cosineSimilarity(queryEmbedding, embedding);

      stepEnd = performance.now();
      searchResults.push({ query: options.query, score: similarity });

      const scoreColor = similarity > 0.5 ? chalk.green.bold : similarity > 0.3 ? chalk.yellow.bold : chalk.dim;
      console.log(scoreColor(`  Similarity: ${(similarity * 100).toFixed(1)}%`));
      console.log(chalk.white(`  Query time: ${formatTime(stepEnd - stepStart)}`));

      benchmarks.push({
        operation: 'Semantic Search',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        success: true,
      });
    } catch (error) {
      stepEnd = performance.now();
      console.log(chalk.red(`  Failed: ${error}`));
      benchmarks.push({
        operation: 'Semantic Search',
        startTime: stepStart,
        endTime: stepEnd,
        durationMs: stepEnd - stepStart,
        success: false,
        error: String(error),
      });
    }
    console.log('');

    // ================================================================
    // STEP 8: Generate Citation
    // ================================================================
    let citation = '';

    if (infAvailable && options.inference !== false) {
      console.log(chalk.white.bold('Step 8: Generate Citation (Inference)'));
      console.log(chalk.dim('-'.repeat(50)));

      gpuBefore = getGpuStats();
      stepStart = performance.now();

      try {
        const citationPrompt = `Based on this document excerpt, generate a proper legal citation:

${cleanText(mergedMarkdown || ocrClean).slice(0, 2000)}

Generate a citation in Bluebook format. Include case name, court, date if available. If this is not a legal case, create an appropriate academic citation.`;

        citation = await runInference(citationPrompt, 'You are a legal citation expert. Generate accurate, properly formatted citations.');

        stepEnd = performance.now();
        gpuAfter = getGpuStats();

        console.log(chalk.green(`  Citation generated in ${formatTime(stepEnd - stepStart)}`));
        console.log(chalk.white.bold(`\n  ${citation.slice(0, 200)}${citation.length > 200 ? '...' : ''}\n`));
        console.log(chalk.cyan(`  ${formatGpuStats(gpuAfter)}`));

        benchmarks.push({
          operation: 'Citation Generation',
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          gpuUtilBefore: gpuBefore?.utilization,
          gpuUtilAfter: gpuAfter?.utilization,
          gpuMemoryMB: gpuAfter?.memoryUsedMB,
          success: true,
        });
      } catch (error) {
        stepEnd = performance.now();
        console.log(chalk.yellow(`  Skipped: ${error}`));
        benchmarks.push({
          operation: 'Citation Generation',
          startTime: stepStart,
          endTime: stepEnd,
          durationMs: stepEnd - stepStart,
          success: false,
          error: String(error),
        });
      }
    } else {
      console.log(chalk.white.bold('Step 8: Citation Generation'));
      console.log(chalk.dim('-'.repeat(50)));
      console.log(chalk.dim('  Skipped (Inference not available)'));
    }
    console.log('');

    // ================================================================
    // BENCHMARK SUMMARY
    // ================================================================
    const endTime = performance.now();
    const totalDuration = endTime - startTime;

    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log(chalk.cyan.bold('                         Benchmark Results'));
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log('');

    console.log(chalk.white.bold('Operation Timings:'));
    console.log(chalk.dim('-'.repeat(50)));

    for (const b of benchmarks) {
      const status = b.success ? chalk.green.bold('[OK]') : chalk.red.bold('[X]');
      const gpu = b.gpuUtilAfter !== undefined ? chalk.cyan.bold(` GPU:${b.gpuUtilAfter}%`) : '';
      console.log(`  ${status} ${chalk.white(b.operation.padEnd(25))} ${chalk.white.bold(formatTime(b.durationMs).padStart(10))}${gpu}`);
    }

    console.log('');
    console.log(chalk.white.bold('Summary:'));
    console.log(chalk.dim('-'.repeat(50)));

    const successOps = benchmarks.filter(b => b.success).length;
    const gpuOps = benchmarks.filter(b => b.gpuUtilAfter !== undefined && b.gpuUtilAfter > 10);
    const avgGpu = gpuOps.length > 0 ? gpuOps.reduce((sum, b) => sum + (b.gpuUtilAfter || 0), 0) / gpuOps.length : 0;
    const maxGpu = Math.max(...benchmarks.map(b => b.gpuUtilAfter || 0));
    const maxMem = Math.max(...benchmarks.map(b => b.gpuMemoryMB || 0));

    console.log(chalk.white(`  Total Duration:      ${chalk.white.bold(formatTime(totalDuration))}`));
    console.log(chalk.white(`  Operations:          ${successOps}/${benchmarks.length} successful`));
    console.log(chalk.white(`  Pages Processed:     ${maxPages}`));
    console.log(chalk.white(`  OCR Characters:      ${ocrText.length}`));
    console.log(chalk.white(`  Embedding Dims:      ${embedding.length}`));
    console.log('');
    console.log(chalk.white.bold('GPU Statistics:'));
    console.log(chalk.dim('-'.repeat(50)));
    console.log(chalk.white(`  Avg Utilization:     ${avgGpu.toFixed(1)}%`));
    console.log(chalk.white(`  Max Utilization:     ${maxGpu}%`));
    console.log(chalk.white(`  Max Memory:          ${maxMem.toFixed(0)} MB`));
    console.log('');

    // Save results
    const results: DemoResults = {
      timestamp: new Date().toISOString(),
      pdfFile: pdfName,
      pageCount: maxPages,
      ocrText,
      extractedText,
      mergedMarkdown,
      embedding,
      searchResults,
      citation,
      benchmarks,
      totalDurationMs: totalDuration,
      gpuStats: {
        avgUtilization: avgGpu,
        maxUtilization: maxGpu,
        maxMemoryMB: maxMem,
      },
    };

    const resultsPath = join(options.output, `demo-results-${Date.now()}.json`);
    writeFileSync(resultsPath, JSON.stringify(results, null, 2));

    console.log(chalk.white(`Results saved: ${resultsPath}`));
    console.log(chalk.white(`Markdown saved: ${mdPath}`));
    console.log('');

    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log(chalk.green.bold('                         Demo Complete!'));
    console.log(chalk.cyan.bold('='.repeat(70)));
    console.log('');
  });

program
  .command('status')
  .description('Show GPU and service status')
  .action(async () => {
    console.log(chalk.blue.bold('\nvLLM Hydra Status\n'));

    // GPU Status
    const gpu = getGpuStats();
    if (gpu) {
      const utilColor = gpu.utilization > 80 ? chalk.red : gpu.utilization > 50 ? chalk.yellow : chalk.green;
      const memPercent = (gpu.memoryUsedMB / gpu.memoryTotalMB * 100);
      const memColor = memPercent > 80 ? chalk.red : memPercent > 50 ? chalk.yellow : chalk.green;

      console.log(chalk.cyan('GPU:'));
      console.log(`  Utilization: ${utilColor(`${gpu.utilization}%`)}`);
      console.log(`  Memory:      ${memColor(`${gpu.memoryUsedMB.toFixed(0)}/${gpu.memoryTotalMB.toFixed(0)} MB (${memPercent.toFixed(0)}%)`)}`);
      console.log(`  Temperature: ${gpu.temperature}C`);
    } else {
      console.log(chalk.gray('GPU: Not available'));
    }
    console.log('');

    // Services
    console.log(chalk.cyan('Services:'));
    const embOk = await checkService(EMBEDDINGS_URL);
    const ocrOk = await checkService(OCR_URL);
    const infOk = await checkService(INFERENCE_URL);

    console.log(`  Embeddings (8001): ${embOk ? chalk.green('Ready') : chalk.red('Not available')}`);
    console.log(`  OCR (8003):        ${ocrOk ? chalk.green('Ready') : chalk.red('Not available')}`);
    console.log(`  Inference (8004):  ${infOk ? chalk.green('Ready') : chalk.yellow('Not available')}`);
    console.log('');
  });

program.parse();
