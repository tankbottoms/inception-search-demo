/**
 * Full OCR Pipeline Demo with Blue Book Citations & GPU Benchmarking
 *
 * Complete document processing pipeline:
 *   1. PDF -> Image conversion
 *   2. HunyuanOCR text extraction
 *   3. pdftotext extraction for comparison
 *   4. ModernBERT embeddings (with load balancer support)
 *   5. Semantic search
 *   6. GPT-OSS 20B for:
 *      - OCR text merging/correction
 *      - Blue Book citation generation
 *      - Document summarization
 *   7. GPU-verified benchmarking
 *
 * Usage:
 *   bun run ocr-pipeline.ts                    # Process all PDFs in files/
 *   bun run ocr-pipeline.ts --pdf path/to.pdf  # Process specific PDF
 *   bun run ocr-pipeline.ts --query "search"   # Search after processing
 *   bun run ocr-pipeline.ts --benchmark        # Run with GPU benchmarks
 */

import axios from 'axios';
import chalk from 'chalk';
import { execSync, spawn } from 'child_process';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { join, basename } from 'path';
import sharp from 'sharp';

// ============================================================
// Configuration
// ============================================================

const EMBEDDINGS_URL = process.env.EMBEDDINGS_URL || 'http://localhost:8001';
const EMBEDDINGS_LB_URL = process.env.EMBEDDINGS_LB_URL || 'http://localhost:8000';
const OCR_URL = process.env.OCR_URL || 'http://localhost:8003';
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://localhost:8004';

const FILES_DIR = process.env.FILES_DIR || join(import.meta.dir, '../../files');
const OUTPUT_DIR = process.env.OUTPUT_DIR || join(import.meta.dir, '../../output');

// ============================================================
// Types
// ============================================================

interface ServiceStatus {
  name: string;
  url: string;
  healthy: boolean;
  model?: string;
  responseTime: number;
}

interface GpuInfo {
  available: boolean;
  name?: string;
  memoryTotal?: string;
  memoryUsed?: string;
  utilization?: string;
  cudaVersion?: string;
}

interface OcrResult {
  page: number;
  ocrText: string;
  ocrTimeMs: number;
}

interface DocumentResult {
  filename: string;
  pageCount: number;
  pdfText: string;
  pdfTextLength: number;
  ocrText: string;
  ocrTextLength: number;
  mergedText: string;
  summary: string;
  bluebookCitation: string;
  embedding: number[];
  timings: {
    pdfToImageMs: number;
    ocrTotalMs: number;
    pdfTextMs: number;
    mergeMs: number;
    summaryMs: number;
    citationMs: number;
    embeddingMs: number;
    totalMs: number;
  };
  quality: {
    ocrConfidence: number;
    textDiff: number;
    improvementRatio: number;
  };
}

interface BenchmarkMetrics {
  gpu: GpuInfo;
  documents: number;
  totalPages: number;
  timings: {
    avgOcrPerPage: number;
    avgEmbedding: number;
    avgMerge: number;
    totalPipeline: number;
  };
  throughput: {
    pagesPerSecond: number;
    tokensPerSecond: number;
  };
}

interface ChatResponse {
  choices: Array<{
    message: {
      content: string | null;
      reasoning_content?: string;
    };
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ============================================================
// GPU Utilities
// ============================================================

function getGpuInfo(): GpuInfo {
  try {
    const output = execSync('nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits', {
      encoding: 'utf-8',
      timeout: 5000,
    });

    const [name, memTotal, memUsed, util, driver] = output.trim().split(',').map(s => s.trim());

    // Get CUDA version
    let cudaVersion = '';
    try {
      const cudaOut = execSync('nvidia-smi --query-gpu=cuda_version --format=csv,noheader', {
        encoding: 'utf-8',
        timeout: 5000,
      });
      cudaVersion = cudaOut.trim();
    } catch {
      cudaVersion = 'N/A';
    }

    return {
      available: true,
      name,
      memoryTotal: `${memTotal} MiB`,
      memoryUsed: `${memUsed} MiB`,
      utilization: `${util}%`,
      cudaVersion,
    };
  } catch {
    return { available: false };
  }
}

function monitorGpuDuringOperation<T>(operation: () => Promise<T>): Promise<{ result: T; peakMemory: number; avgUtilization: number }> {
  return new Promise(async (resolve, reject) => {
    const samples: { memory: number; utilization: number }[] = [];
    let running = true;

    // Background GPU sampling
    const sampler = setInterval(() => {
      if (!running) return;
      try {
        const output = execSync('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits', {
          encoding: 'utf-8',
          timeout: 1000,
        });
        const [mem, util] = output.trim().split(',').map(s => parseFloat(s.trim()));
        samples.push({ memory: mem, utilization: util });
      } catch {
        // Ignore sampling errors
      }
    }, 100);

    try {
      const result = await operation();
      running = false;
      clearInterval(sampler);

      const peakMemory = samples.length > 0 ? Math.max(...samples.map(s => s.memory)) : 0;
      const avgUtilization = samples.length > 0 ? samples.reduce((a, b) => a + b.utilization, 0) / samples.length : 0;

      resolve({ result, peakMemory, avgUtilization });
    } catch (error) {
      running = false;
      clearInterval(sampler);
      reject(error);
    }
  });
}

// ============================================================
// Service Clients
// ============================================================

async function checkService(url: string, name: string): Promise<ServiceStatus> {
  const start = performance.now();
  try {
    await axios.get(`${url}/health`, { timeout: 10000 });
    const modelRes = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    const model = modelRes.data.data?.[0]?.id;
    return { name, url, healthy: true, model, responseTime: performance.now() - start };
  } catch (error) {
    return { name, url, healthy: false, responseTime: performance.now() - start };
  }
}

async function getModelId(url: string): Promise<string | null> {
  try {
    const response = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    return response.data.data?.[0]?.id || null;
  } catch {
    return null;
  }
}

// ============================================================
// PDF Processing
// ============================================================

async function extractPdfTextNative(pdfPath: string): Promise<{ text: string; timeMs: number }> {
  const start = performance.now();

  try {
    // Try pdftotext first (most accurate)
    const text = execSync(`pdftotext -layout "${pdfPath}" -`, {
      encoding: 'utf-8',
      timeout: 30000,
    });
    return { text: text.trim(), timeMs: performance.now() - start };
  } catch {
    // Fallback to pdf-parse
    try {
      const pdfParse = (await import('pdf-parse')).default;
      const buffer = readFileSync(pdfPath);
      const result = await pdfParse(buffer);
      return { text: result.text.trim(), timeMs: performance.now() - start };
    } catch {
      return { text: '', timeMs: performance.now() - start };
    }
  }
}

async function pdfToImages(pdfPath: string, outputDir: string): Promise<{ paths: string[]; timeMs: number }> {
  const start = performance.now();
  const { fromPath } = await import('pdf2pic');

  const filename = basename(pdfPath, '.pdf');
  const imageDir = join(outputDir, filename);
  mkdirSync(imageDir, { recursive: true });

  const options = {
    density: 200,
    saveFilename: 'page',
    savePath: imageDir,
    format: 'png',
    width: 1600,
    height: 2200,
  };

  const convert = fromPath(pdfPath, options);

  // Get page count
  const pdfParse = (await import('pdf-parse')).default;
  const buffer = readFileSync(pdfPath);
  const { numpages } = await pdfParse(buffer);

  const paths: string[] = [];
  for (let i = 1; i <= numpages; i++) {
    try {
      const result = await convert(i);
      if (result.path) paths.push(result.path);
    } catch {
      console.log(chalk.yellow(`    Warning: Could not convert page ${i}`));
    }
  }

  return { paths, timeMs: performance.now() - start };
}

async function imageToBase64(imagePath: string): Promise<string> {
  const buffer = await sharp(imagePath)
    .resize(2400, 3200, { fit: 'inside', withoutEnlargement: true })
    .png({ quality: 90 })
    .toBuffer();
  return buffer.toString('base64');
}

// ============================================================
// OCR Service
// ============================================================

async function runOcrOnImage(imageBase64: string, modelId: string): Promise<string> {
  const response = await axios.post<ChatResponse>(
    `${OCR_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Extract ALL text from this document image. Preserve the original formatting, paragraphs, and structure as much as possible. Include all headers, footnotes, page numbers, and any visible text. Output only the extracted text with no commentary.`,
            },
            {
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${imageBase64}` },
            },
          ],
        },
      ],
      max_tokens: 8192,
      temperature: 0.1,
    },
    { timeout: 300000 }
  );

  return response.data.choices[0].message.content || '';
}

async function processOcrPages(imagePaths: string[], modelId: string): Promise<{ ocrText: string; results: OcrResult[]; totalMs: number }> {
  const results: OcrResult[] = [];
  const overallStart = performance.now();

  for (let i = 0; i < imagePaths.length; i++) {
    const pageStart = performance.now();
    console.log(chalk.gray(`      OCR page ${i + 1}/${imagePaths.length}...`));

    const imageBase64 = await imageToBase64(imagePaths[i]);
    const ocrText = await runOcrOnImage(imageBase64, modelId);

    results.push({
      page: i + 1,
      ocrText,
      ocrTimeMs: performance.now() - pageStart,
    });
  }

  const ocrText = results.map((r, i) =>
    `\n--- Page ${i + 1} ---\n\n${r.ocrText}`
  ).join('\n');

  return {
    ocrText,
    results,
    totalMs: performance.now() - overallStart,
  };
}

// ============================================================
// GPT-OSS Text Processing
// ============================================================

async function mergeAndCorrectText(
  ocrText: string,
  pdfText: string,
  modelId: string
): Promise<{ mergedText: string; timeMs: number }> {
  const start = performance.now();

  const prompt = `You are a document text correction assistant. You have two versions of the same document:

1. OCR TEXT (extracted via vision model from images):
---
${ocrText.slice(0, 6000)}
---

2. PDF TEXT (extracted directly from PDF):
---
${pdfText.slice(0, 6000)}
---

Your task:
1. Merge these two sources to create the most accurate representation
2. Fix any OCR errors (misspellings, wrong characters, broken words)
3. Correct formatting issues (random spaces, line breaks in wrong places)
4. Preserve the document structure (paragraphs, headers, lists)
5. Remove any duplicate text or artifacts
6. Keep legal citations and case names accurate

Output ONLY the corrected, merged text in clean Markdown format. No commentary.`;

  const response = await axios.post<ChatResponse>(
    `${INFERENCE_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 4096,
      temperature: 0.2,
    },
    { timeout: 120000 }
  );

  return {
    mergedText: response.data.choices[0].message.content || ocrText,
    timeMs: performance.now() - start,
  };
}

async function generateSummary(text: string, modelId: string): Promise<{ summary: string; timeMs: number }> {
  const start = performance.now();

  const prompt = `Summarize the following legal document in 2-3 concise paragraphs. Focus on:
- The main parties involved
- The key legal issues or claims
- The outcome or current status
- Any significant legal principles established

Document:
---
${text.slice(0, 8000)}
---

Provide a clear, professional summary:`;

  const response = await axios.post<ChatResponse>(
    `${INFERENCE_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 1024,
      temperature: 0.3,
    },
    { timeout: 60000 }
  );

  return {
    summary: response.data.choices[0].message.content || '',
    timeMs: performance.now() - start,
  };
}

async function generateBluebookCitation(
  text: string,
  filename: string,
  modelId: string
): Promise<{ citation: string; timeMs: number; reasoning: string }> {
  const start = performance.now();

  const prompt = `Based on the following legal document, generate a proper Blue Book citation.

Filename: ${filename}

Document text (first 4000 chars):
---
${text.slice(0, 4000)}
---

Identify from the text:
1. Case name (parties)
2. Court
3. Volume and reporter
4. Page number
5. Year decided

Then provide the citation in proper Blue Book format. If this is not a case, provide the appropriate citation format (statute, regulation, secondary source, etc.).

Output format:
CITATION: [the Blue Book citation]
TYPE: [case/statute/regulation/secondary]
NOTES: [any relevant notes about the citation]`;

  const response = await axios.post<ChatResponse>(
    `${INFERENCE_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 512,
      temperature: 0.2,
    },
    { timeout: 60000 }
  );

  const content = response.data.choices[0].message.content || '';
  const reasoning = response.data.choices[0].message.reasoning_content || '';

  // Extract citation from response
  const citationMatch = content.match(/CITATION:\s*(.+?)(?:\n|TYPE:|$)/s);
  const citation = citationMatch ? citationMatch[1].trim() : content.split('\n')[0];

  return {
    citation,
    timeMs: performance.now() - start,
    reasoning,
  };
}

// ============================================================
// Embeddings
// ============================================================

async function generateEmbedding(
  text: string,
  useLB: boolean = false
): Promise<{ embedding: number[]; timeMs: number; endpoint: string }> {
  const start = performance.now();
  const endpoint = useLB ? EMBEDDINGS_LB_URL : EMBEDDINGS_URL;
  const modelId = await getModelId(endpoint);

  if (!modelId) throw new Error(`Embeddings model not available at ${endpoint}`);

  // Truncate text for 512-token model
  const truncatedText = text.slice(0, 2000);

  const response = await axios.post(
    `${endpoint}/v1/embeddings`,
    {
      model: modelId,
      input: truncatedText,
    },
    { timeout: 30000 }
  );

  return {
    embedding: response.data.data[0].embedding,
    timeMs: performance.now() - start,
    endpoint,
  };
}

async function generateBatchEmbeddings(
  texts: string[],
  useLB: boolean = false
): Promise<{ embeddings: number[][]; timeMs: number; rps: number }> {
  const start = performance.now();
  const endpoint = useLB ? EMBEDDINGS_LB_URL : EMBEDDINGS_URL;
  const modelId = await getModelId(endpoint);

  if (!modelId) throw new Error(`Embeddings model not available at ${endpoint}`);

  // Truncate all texts
  const truncatedTexts = texts.map(t => t.slice(0, 2000));

  const response = await axios.post(
    `${endpoint}/v1/embeddings`,
    {
      model: modelId,
      input: truncatedTexts,
    },
    { timeout: 60000 }
  );

  const timeMs = performance.now() - start;
  return {
    embeddings: response.data.data.map((d: any) => d.embedding),
    timeMs,
    rps: (texts.length / timeMs) * 1000,
  };
}

// ============================================================
// Text Comparison
// ============================================================

function calculateTextDifference(text1: string, text2: string): number {
  // Simple word-level Jaccard similarity
  const words1 = new Set(text1.toLowerCase().match(/\b\w+\b/g) || []);
  const words2 = new Set(text2.toLowerCase().match(/\b\w+\b/g) || []);

  const intersection = new Set([...words1].filter(w => words2.has(w)));
  const union = new Set([...words1, ...words2]);

  return union.size > 0 ? intersection.size / union.size : 0;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============================================================
// Report Generation
// ============================================================

function generateMarkdownReport(doc: DocumentResult): string {
  return `# Document Analysis Report

## File: ${doc.filename}
**Pages:** ${doc.pageCount}
**Generated:** ${new Date().toISOString()}

---

## Blue Book Citation

\`\`\`
${doc.bluebookCitation}
\`\`\`

---

## Summary

${doc.summary}

---

## Text Extraction Comparison

| Source | Characters | Quality |
|--------|-----------|---------|
| PDF Native (pdftotext) | ${doc.pdfTextLength.toLocaleString()} | Baseline |
| OCR (HunyuanOCR) | ${doc.ocrTextLength.toLocaleString()} | ${(doc.quality.ocrConfidence * 100).toFixed(1)}% match |
| Merged (GPT-OSS) | ${doc.mergedText.length.toLocaleString()} | ${(doc.quality.improvementRatio * 100).toFixed(1)}% improvement |

### Text Similarity
- OCR vs PDF: **${(doc.quality.textDiff * 100).toFixed(1)}%** word overlap

---

## Processing Timings

| Stage | Duration |
|-------|----------|
| PDF to Images | ${doc.timings.pdfToImageMs.toFixed(0)}ms |
| OCR (all pages) | ${doc.timings.ocrTotalMs.toFixed(0)}ms |
| PDF Text Extract | ${doc.timings.pdfTextMs.toFixed(0)}ms |
| Text Merge | ${doc.timings.mergeMs.toFixed(0)}ms |
| Summary | ${doc.timings.summaryMs.toFixed(0)}ms |
| Citation | ${doc.timings.citationMs.toFixed(0)}ms |
| Embedding | ${doc.timings.embeddingMs.toFixed(0)}ms |
| **Total** | **${doc.timings.totalMs.toFixed(0)}ms** |

---

## Merged Document Text

${doc.mergedText.slice(0, 10000)}

${doc.mergedText.length > 10000 ? `\n\n... (${doc.mergedText.length - 10000} more characters)` : ''}
`;
}

// ============================================================
// Main Pipeline
// ============================================================

async function processDocument(
  pdfPath: string,
  services: { ocrModelId: string; infModelId: string; useLB: boolean }
): Promise<DocumentResult> {
  const filename = basename(pdfPath);
  const pipelineStart = performance.now();

  console.log(chalk.white.bold(`\n  Processing: ${filename}`));

  // Step 1: Convert PDF to images
  console.log(chalk.gray('    [1/7] Converting PDF to images...'));
  const imagesDir = join(OUTPUT_DIR, 'images');
  mkdirSync(imagesDir, { recursive: true });
  const { paths: imagePaths, timeMs: pdfToImageMs } = await pdfToImages(pdfPath, imagesDir);
  console.log(chalk.green(`    OK ${imagePaths.length} pages in ${pdfToImageMs.toFixed(0)}ms`));

  // Step 2: Extract text with pdftotext
  console.log(chalk.gray('    [2/7] Extracting PDF text (pdftotext)...'));
  const { text: pdfText, timeMs: pdfTextMs } = await extractPdfTextNative(pdfPath);
  console.log(chalk.green(`    OK ${pdfText.length} chars in ${pdfTextMs.toFixed(0)}ms`));

  // Step 3: OCR all pages
  console.log(chalk.gray('    [3/7] Running OCR on all pages...'));
  const { ocrText, totalMs: ocrTotalMs } = await processOcrPages(imagePaths, services.ocrModelId);
  console.log(chalk.green(`    OK ${ocrText.length} chars in ${ocrTotalMs.toFixed(0)}ms`));

  // Step 4: Merge and correct text
  console.log(chalk.gray('    [4/7] Merging and correcting text...'));
  const { mergedText, timeMs: mergeMs } = await mergeAndCorrectText(
    ocrText,
    pdfText,
    services.infModelId
  );
  console.log(chalk.green(`    OK ${mergedText.length} chars in ${mergeMs.toFixed(0)}ms`));

  // Step 5: Generate summary
  console.log(chalk.gray('    [5/7] Generating summary...'));
  const { summary, timeMs: summaryMs } = await generateSummary(mergedText, services.infModelId);
  console.log(chalk.green(`    OK ${summary.length} chars in ${summaryMs.toFixed(0)}ms`));

  // Step 6: Generate Blue Book citation
  console.log(chalk.gray('    [6/7] Generating Blue Book citation...'));
  const { citation, timeMs: citationMs, reasoning } = await generateBluebookCitation(
    mergedText,
    filename,
    services.infModelId
  );
  console.log(chalk.green(`    OK in ${citationMs.toFixed(0)}ms`));
  console.log(chalk.cyan(`    Citation: ${citation}`));

  // Step 7: Generate embedding
  console.log(chalk.gray('    [7/7] Generating embedding...'));
  const { embedding, timeMs: embeddingMs, endpoint } = await generateEmbedding(
    mergedText,
    services.useLB
  );
  console.log(chalk.green(`    OK ${embedding.length} dims in ${embeddingMs.toFixed(0)}ms (${endpoint})`));

  // Calculate quality metrics
  const textDiff = calculateTextDifference(ocrText, pdfText);
  const ocrConfidence = Math.min(1, ocrText.length / Math.max(pdfText.length, 1));
  const improvementRatio = mergedText.length > 0
    ? Math.min(2, mergedText.length / Math.max(pdfText.length, ocrText.length, 1))
    : 0;

  // Cleanup images
  for (const imgPath of imagePaths) {
    try { unlinkSync(imgPath); } catch {}
  }

  const totalMs = performance.now() - pipelineStart;

  return {
    filename,
    pageCount: imagePaths.length,
    pdfText,
    pdfTextLength: pdfText.length,
    ocrText,
    ocrTextLength: ocrText.length,
    mergedText,
    summary,
    bluebookCitation: citation,
    embedding,
    timings: {
      pdfToImageMs,
      ocrTotalMs,
      pdfTextMs,
      mergeMs,
      summaryMs,
      citationMs,
      embeddingMs,
      totalMs,
    },
    quality: {
      ocrConfidence,
      textDiff,
      improvementRatio,
    },
  };
}

// ============================================================
// CLI Entry Point
// ============================================================

async function main() {
  const args = process.argv.slice(2);

  console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
  console.log(chalk.cyan.bold('     OCR Pipeline with Blue Book Citations & GPU Benchmarking'));
  console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

  // Parse arguments
  const pdfIdx = args.indexOf('--pdf');
  const specificPdf = pdfIdx !== -1 ? args[pdfIdx + 1] : null;
  const queryIdx = args.indexOf('--query');
  const searchQuery = queryIdx !== -1 ? args[queryIdx + 1] : null;
  const runBenchmark = args.includes('--benchmark');
  const useLB = args.includes('--load-balanced') || args.includes('--lb');

  // Check GPU
  console.log(chalk.white.bold('GPU Status'));
  console.log(chalk.dim('-'.repeat(50)));
  const gpuInfo = getGpuInfo();
  if (gpuInfo.available) {
    console.log(chalk.green(`  GPU: ${gpuInfo.name}`));
    console.log(chalk.gray(`  Memory: ${gpuInfo.memoryUsed} / ${gpuInfo.memoryTotal}`));
    console.log(chalk.gray(`  Utilization: ${gpuInfo.utilization}`));
    console.log(chalk.gray(`  CUDA: ${gpuInfo.cudaVersion}`));
  } else {
    console.log(chalk.yellow('  GPU: Not available (CPU mode)'));
  }
  console.log('');

  // Check services
  console.log(chalk.white.bold('Service Discovery'));
  console.log(chalk.dim('-'.repeat(50)));

  const embStatus = await checkService(EMBEDDINGS_URL, 'Embeddings');
  const embLbStatus = await checkService(EMBEDDINGS_LB_URL, 'Embeddings LB');
  const ocrStatus = await checkService(OCR_URL, 'OCR');
  const infStatus = await checkService(INFERENCE_URL, 'Inference');

  console.log(embStatus.healthy
    ? chalk.green(`  [OK] Embeddings (${EMBEDDINGS_URL})`) + chalk.dim(` - ${embStatus.model}`)
    : chalk.red(`  [X] Embeddings: not available`));
  console.log(embLbStatus.healthy
    ? chalk.green(`  [OK] Embeddings LB (${EMBEDDINGS_LB_URL})`) + chalk.dim(` - Load balanced`)
    : chalk.dim(`  [--] Embeddings LB: not available`));
  console.log(ocrStatus.healthy
    ? chalk.green(`  [OK] OCR (${OCR_URL})`) + chalk.dim(` - ${ocrStatus.model}`)
    : chalk.red(`  [X] OCR: not available`));
  console.log(infStatus.healthy
    ? chalk.green(`  [OK] Inference (${INFERENCE_URL})`) + chalk.dim(` - ${infStatus.model}`)
    : chalk.red(`  [X] Inference: not available`));
  console.log('');

  // Validate required services
  if (!ocrStatus.healthy || !infStatus.healthy || !embStatus.healthy) {
    console.log(chalk.red('Required services not available. Start the vLLM Hydra cluster:'));
    console.log(chalk.gray('  docker compose --profile gpt-oss up -d'));
    process.exit(1);
  }

  // Get PDF files
  let pdfFiles: string[] = [];
  if (specificPdf) {
    if (!existsSync(specificPdf)) {
      console.log(chalk.red(`PDF not found: ${specificPdf}`));
      process.exit(1);
    }
    pdfFiles = [specificPdf];
  } else {
    if (!existsSync(FILES_DIR)) {
      console.log(chalk.red(`Files directory not found: ${FILES_DIR}`));
      console.log(chalk.gray('  Create the directory and add PDF files'));
      process.exit(1);
    }
    pdfFiles = readdirSync(FILES_DIR)
      .filter(f => f.endsWith('.pdf'))
      .map(f => join(FILES_DIR, f));
  }

  if (pdfFiles.length === 0) {
    console.log(chalk.yellow('No PDF files found'));
    console.log(chalk.gray(`  Add PDFs to: ${FILES_DIR}`));
    process.exit(1);
  }

  console.log(chalk.white.bold(`Processing ${pdfFiles.length} PDF(s)`));
  console.log(chalk.dim('-'.repeat(50)));

  const services = {
    ocrModelId: ocrStatus.model!,
    infModelId: infStatus.model!,
    useLB: useLB && embLbStatus.healthy,
  };

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const results: DocumentResult[] = [];
  const overallStart = performance.now();

  for (const pdfPath of pdfFiles) {
    try {
      const result = await processDocument(pdfPath, services);
      results.push(result);

      // Save individual report
      const reportPath = join(OUTPUT_DIR, `${basename(pdfPath, '.pdf')}-report.md`);
      writeFileSync(reportPath, generateMarkdownReport(result));
      console.log(chalk.gray(`    Report saved: ${reportPath}`));

    } catch (error) {
      console.log(chalk.red(`    Error: ${error instanceof Error ? error.message : error}`));
    }
  }

  const overallTime = performance.now() - overallStart;

  // Summary
  console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
  console.log(chalk.cyan.bold('                       Pipeline Summary'));
  console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

  const totalPages = results.reduce((sum, r) => sum + r.pageCount, 0);
  const avgOcrPerPage = results.length > 0
    ? results.reduce((sum, r) => sum + r.timings.ocrTotalMs / r.pageCount, 0) / results.length
    : 0;

  console.log(chalk.white.bold('Documents Processed:'));
  for (const r of results) {
    console.log(`  ${chalk.green(r.filename)}`);
    console.log(chalk.gray(`    Pages: ${r.pageCount} | Citation: ${r.bluebookCitation.slice(0, 60)}...`));
  }
  console.log('');

  console.log(chalk.white.bold('Performance Metrics:'));
  console.log(chalk.gray(`  Documents: ${results.length}`));
  console.log(chalk.gray(`  Total pages: ${totalPages}`));
  console.log(chalk.gray(`  Total time: ${(overallTime / 1000).toFixed(1)}s`));
  console.log(chalk.gray(`  Avg OCR/page: ${avgOcrPerPage.toFixed(0)}ms`));
  console.log(chalk.gray(`  Throughput: ${(totalPages / (overallTime / 1000)).toFixed(2)} pages/sec`));

  // GPU benchmark summary
  if (runBenchmark && gpuInfo.available) {
    console.log('');
    console.log(chalk.white.bold('GPU Benchmark:'));
    const finalGpu = getGpuInfo();
    console.log(chalk.gray(`  Peak memory: ${finalGpu.memoryUsed}`));
    console.log(chalk.gray(`  Final utilization: ${finalGpu.utilization}`));
  }

  // Search if query provided
  if (searchQuery && results.length > 0) {
    console.log('');
    console.log(chalk.cyan.bold('='.repeat(80)));
    console.log(chalk.cyan.bold('                       Semantic Search'));
    console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

    console.log(chalk.white(`Query: "${searchQuery}"\n`));

    const { embedding: queryEmb } = await generateEmbedding(searchQuery, services.useLB);

    const searchResults = results.map(r => ({
      filename: r.filename,
      score: cosineSimilarity(queryEmb, r.embedding),
      citation: r.bluebookCitation,
      summary: r.summary.slice(0, 200),
    })).sort((a, b) => b.score - a.score);

    for (let i = 0; i < searchResults.length; i++) {
      const r = searchResults[i];
      const scoreColor = r.score > 0.7 ? chalk.green : r.score > 0.5 ? chalk.yellow : chalk.gray;
      console.log(`${i + 1}. ${scoreColor(`${(r.score * 100).toFixed(1)}%`)} - ${r.filename}`);
      console.log(chalk.cyan(`   ${r.citation}`));
      console.log(chalk.gray(`   ${r.summary}...`));
      console.log('');
    }
  }

  // Save combined results
  const resultsPath = join(OUTPUT_DIR, 'pipeline-results.json');
  writeFileSync(resultsPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    gpu: gpuInfo,
    documents: results.map(r => ({
      filename: r.filename,
      pageCount: r.pageCount,
      bluebookCitation: r.bluebookCitation,
      summary: r.summary,
      timings: r.timings,
      quality: r.quality,
    })),
    metrics: {
      totalDocuments: results.length,
      totalPages,
      totalTimeMs: overallTime,
      avgOcrPerPageMs: avgOcrPerPage,
    },
  }, null, 2));

  console.log('');
  console.log(chalk.gray(`Results saved: ${resultsPath}`));
  console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
  console.log(chalk.green.bold('                    Pipeline Complete'));
  console.log(chalk.cyan.bold('='.repeat(80) + '\n'));
}

main().catch(error => {
  console.log(chalk.red(`\nFatal error: ${error.message}`));
  console.error(error);
  process.exit(1);
});
