/**
 * Comprehensive Benchmark - OCR + Embedding + Citation + Summary Pipeline
 *
 * Full pipeline benchmark with detailed metrics:
 *   1. PDF -> Image conversion
 *   2. HunyuanOCR text extraction (parallel across nodes)
 *   3. ModernBERT embedding generation
 *   4. GPT-OSS Blue Book citation generation (any document type)
 *   5. GPT-OSS document summarization
 *
 * Usage:
 *   bun run stability-test.ts                     # Single run forward
 *   bun run stability-test.ts --iterations 2     # Multiple iterations
 *   bun run stability-test.ts --reverse          # Run in reverse order
 *   bun run stability-test.ts --iterations 2 --forward-backward  # 2x forward + 2x backward
 */

import axios from 'axios';
import chalk from 'chalk';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, unlinkSync, rmSync, statSync } from 'fs';
import { join, basename } from 'path';
import sharp from 'sharp';

// ============================================================
// Configuration
// ============================================================

const EMBEDDINGS_URL = process.env.EMBEDDINGS_URL || 'http://localhost:8001';
const EMBEDDINGS_LB_URL = process.env.EMBEDDINGS_LB_URL || 'http://localhost:8000';
const OCR_URL = process.env.OCR_URL || 'http://localhost:8003';
const SPARK2_IP = '192.168.1.63';

const INFERENCE_URLS = [
  process.env.INFERENCE_URL || 'http://localhost:8004',
  `http://${SPARK2_IP}:8004`,
];

const FILES_DIR = process.env.FILES_DIR || join(import.meta.dir, '../../demo/files');
const OUTPUT_DIR = process.env.OUTPUT_DIR || join(import.meta.dir, '../../output');

// ============================================================
// Types
// ============================================================

interface DocumentResult {
  filename: string;
  fileSizeBytes: number;
  pageCount: number;
  totalChars: number;
  charsPerPage: number;
  status: 'success' | 'failed';
  ocrStatus: 'pass' | 'fail' | 'skip';
  embeddingStatus: 'pass' | 'fail' | 'skip';
  citationStatus: 'pass' | 'fail' | 'skip';
  summaryStatus: 'pass' | 'fail' | 'skip';
  ocrText: string;
  citation: string;
  summary: string;
  embedding: number[];
  error?: string;
  timings: {
    pdfToImageMs: number;
    ocrMs: number;
    ocrPerPageMs: number;
    embeddingMs: number;
    citationMs: number;
    summaryMs: number;
    totalMs: number;
  };
}

interface IterationResult {
  iteration: number;
  direction: 'forward' | 'backward';
  documents: DocumentResult[];
  metrics: {
    total: number;
    passed: number;
    failed: number;
    ocrPassed: number;
    embeddingPassed: number;
    citationPassed: number;
    summaryPassed: number;
    totalPages: number;
    totalChars: number;
    avgCharsPerPage: number;
    avgOcrPerPageMs: number;
    avgOcrPerDocMs: number;
    avgEmbeddingMs: number;
    avgCitationMs: number;
    avgSummaryMs: number;
    avgTotalPerDocMs: number;
    totalTimeMs: number;
    docsPerMinute: number;
    pagesPerMinute: number;
  };
}

interface BenchmarkReport {
  timestamp: string;
  config: {
    iterations: number;
    forwardBackward: boolean;
    pdfCount: number;
    filesDir: string;
  };
  services: {
    ocrEndpoints: number;
    embeddingsOk: boolean;
    inferenceOk: boolean;
  };
  iterations: IterationResult[];
  aggregate: {
    totalDocs: number;
    totalPages: number;
    totalChars: number;
    passRate: number;
    avgOcrPerPageMs: number;
    avgOcrPerDocMs: number;
    avgEmbeddingMs: number;
    avgCitationMs: number;
    avgSummaryMs: number;
    avgTotalPerDocMs: number;
    docsPerMinute: number;
    pagesPerMinute: number;
  };
  projections: {
    docs100: { estimatedMinutes: number; estimatedHours: number };
    docs1000: { estimatedMinutes: number; estimatedHours: number };
  };
  verdict: 'PASSED' | 'FAILED';
}

interface ChatResponse {
  choices: Array<{
    message: {
      content: string | null;
      reasoning_content?: string;
    };
    finish_reason?: string;
  }>;
}

// ============================================================
// Service Clients
// ============================================================

async function checkService(url: string): Promise<{ healthy: boolean; model?: string }> {
  try {
    await axios.get(`${url}/health`, { timeout: 5000 });
    const modelRes = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    const model = modelRes.data.data?.[0]?.id;
    return { healthy: true, model };
  } catch {
    return { healthy: false };
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

async function pdfToImages(pdfPath: string, outputDir: string): Promise<{ paths: string[]; timeMs: number }> {
  const start = performance.now();
  const { fromPath } = await import('pdf2pic');

  const filename = basename(pdfPath, '.pdf');
  const imageDir = join(outputDir, filename);

  if (existsSync(imageDir)) {
    rmSync(imageDir, { recursive: true, force: true });
  }
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

  const pdfParse = (await import('pdf-parse')).default;
  const buffer = readFileSync(pdfPath);
  const { numpages } = await pdfParse(buffer);

  const paths: string[] = [];
  for (let i = 1; i <= numpages; i++) {
    try {
      const result = await convert(i);
      if (result.path) paths.push(result.path);
    } catch {
      // Skip failed pages
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
// OCR Service (Multi-Node Parallel)
// ============================================================

interface OcrEndpoint {
  url: string;
  host: string;
  modelId: string;
}

async function getAvailableOcrEndpoints(): Promise<OcrEndpoint[]> {
  const endpoints: OcrEndpoint[] = [];
  const ocrUrls = [
    { url: OCR_URL, host: 'spark-1' },
    { url: `http://${SPARK2_IP}:8003`, host: 'spark-2' },
  ];

  for (const ep of ocrUrls) {
    try {
      const status = await checkService(ep.url);
      if (status.healthy && status.model) {
        endpoints.push({ url: ep.url, host: ep.host, modelId: status.model });
      }
    } catch {
      // Skip unavailable
    }
  }

  return endpoints;
}

async function runOcrOnImage(imageBase64: string, modelId: string, ocrUrl: string): Promise<string> {
  const response = await axios.post<ChatResponse>(
    `${ocrUrl}/v1/chat/completions`,
    {
      model: modelId,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Extract ALL text from this document image. Preserve formatting and structure. Output only the extracted text.',
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

async function processOcrParallel(imagePaths: string[]): Promise<{ ocrText: string; timeMs: number }> {
  const start = performance.now();
  const endpoints = await getAvailableOcrEndpoints();

  if (endpoints.length === 0) {
    throw new Error('No OCR endpoints available');
  }

  const concurrency = endpoints.length * 2;
  const results: string[] = new Array(imagePaths.length);
  const inFlight: Promise<void>[] = [];
  let taskIndex = 0;

  const processPage = async (pageNum: number, imagePath: string, endpoint: OcrEndpoint): Promise<void> => {
    const imageBase64 = await imageToBase64(imagePath);
    const text = await runOcrOnImage(imageBase64, endpoint.modelId, endpoint.url);
    results[pageNum] = text;
  };

  while (taskIndex < imagePaths.length || inFlight.length > 0) {
    while (inFlight.length < concurrency && taskIndex < imagePaths.length) {
      const pageNum = taskIndex;
      const imagePath = imagePaths[taskIndex];
      const endpoint = endpoints[taskIndex % endpoints.length];
      taskIndex++;

      const promise = processPage(pageNum, imagePath, endpoint).then(() => {
        const idx = inFlight.indexOf(promise);
        if (idx > -1) inFlight.splice(idx, 1);
      });
      inFlight.push(promise);
    }

    if (inFlight.length > 0) {
      await Promise.race(inFlight);
    }
  }

  const ocrText = results.map((text, i) => `\n--- Page ${i + 1} ---\n\n${text}`).join('\n');
  return { ocrText, timeMs: performance.now() - start };
}

// ============================================================
// Embedding Service
// ============================================================

async function generateEmbedding(text: string): Promise<{ embedding: number[]; timeMs: number }> {
  const start = performance.now();
  const modelId = await getModelId(EMBEDDINGS_LB_URL) || await getModelId(EMBEDDINGS_URL);

  if (!modelId) throw new Error('No embedding model available');

  const truncatedText = text.slice(0, 1500);
  const endpoint = (await checkService(EMBEDDINGS_LB_URL)).healthy ? EMBEDDINGS_LB_URL : EMBEDDINGS_URL;

  const response = await axios.post(
    `${endpoint}/v1/embeddings`,
    { model: modelId, input: truncatedText },
    { timeout: 30000 }
  );

  return {
    embedding: response.data.data[0].embedding,
    timeMs: performance.now() - start,
  };
}

// ============================================================
// Inference (Citation + Summary)
// ============================================================

let activeInferenceUrl: string | null = null;

async function findInferenceEndpoint(): Promise<string | null> {
  for (const url of INFERENCE_URLS) {
    const status = await checkService(url);
    if (status.healthy) {
      activeInferenceUrl = url;
      return url;
    }
  }
  return null;
}

async function generateCitation(ocrText: string, filename: string): Promise<{ citation: string; timeMs: number }> {
  const start = performance.now();

  if (!activeInferenceUrl) {
    throw new Error('No inference endpoint available');
  }

  const modelId = await getModelId(activeInferenceUrl);
  if (!modelId) throw new Error('No inference model available');

  // Expanded prompt for any document type with Blue Book citation format
  const prompt = `Generate a proper Blue Book citation for this document. The Blue Book citation format varies by document type:

**Document types and formats:**
- Court cases: Party v. Party, Volume Reporter Page (Court Year)
- Statutes: Title U.S.C. Section (Year)
- Books: Author, Title Page (Publisher Year)
- Articles/Papers: Author, Title, Volume Journal Page (Year)
- Reports: Author/Org, Title (Report No., Year)
- SEC Filings: Company Name, Form Type (Filing Date)
- Resumes/CVs: Name, Resume/CV (Date if available)
- Websites/Online: Author, Title, URL (last visited Date)
- Miscellaneous: Title/Description (Date or "n.d.")

**Filename:** ${filename}

**Document text (first 3000 chars):**
---
${ocrText.slice(0, 3000)}
---

**Instructions:**
1. Identify the document type from the content
2. Extract: author/parties, title, date, source/court/publisher
3. If date unknown, use the filename hash or "n.d."
4. Output ONLY the Blue Book citation on a single line

Citation:`;

  const response = await axios.post<ChatResponse>(
    `${activeInferenceUrl}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 512,
      temperature: 0.2,
    },
    { timeout: 90000 }
  );

  let citation = response.data.choices[0].message.content?.trim() || '';

  // If content is empty, extract from reasoning_content
  if (!citation && response.data.choices[0].message.reasoning_content) {
    const reasoning = response.data.choices[0].message.reasoning_content;
    // Try to find any citation-like text
    const lines = reasoning.split('\n').filter(l => l.trim().length > 20);
    if (lines.length > 0) {
      // Take the last substantive line as likely the citation
      citation = lines[lines.length - 1].trim();
    }
  }

  // Fallback: create minimal citation from filename
  if (!citation || citation.length < 10) {
    const cleanName = filename.replace('.pdf', '').replace(/[_-]/g, ' ');
    citation = `Document: ${cleanName} (${new Date().getFullYear()})`;
  }

  return { citation, timeMs: performance.now() - start };
}

async function generateSummary(ocrText: string): Promise<{ summary: string; timeMs: number }> {
  const start = performance.now();

  if (!activeInferenceUrl) {
    throw new Error('No inference endpoint available');
  }

  const modelId = await getModelId(activeInferenceUrl);
  if (!modelId) throw new Error('No inference model available');

  const prompt = `Summarize this document in 2-3 sentences. Focus on the main topic, parties involved, and key points.

Document text (first 2500 chars):
---
${ocrText.slice(0, 2500)}
---

Summary:`;

  const response = await axios.post<ChatResponse>(
    `${activeInferenceUrl}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 256,
      temperature: 0.3,
    },
    { timeout: 60000 }
  );

  let summary = response.data.choices[0].message.content?.trim() || '';

  // If content is empty, try reasoning_content
  if (!summary && response.data.choices[0].message.reasoning_content) {
    const reasoning = response.data.choices[0].message.reasoning_content;
    // Extract summary from reasoning
    const sentences = reasoning.split(/[.!?]+/).filter(s => s.trim().length > 30);
    if (sentences.length >= 2) {
      summary = sentences.slice(-2).join('. ').trim() + '.';
    }
  }

  return { summary: summary || 'Summary not available', timeMs: performance.now() - start };
}

// ============================================================
// Document Processing
// ============================================================

async function processDocument(pdfPath: string): Promise<DocumentResult> {
  const filename = basename(pdfPath);
  const docStart = performance.now();

  let fileSizeBytes = 0;
  try {
    fileSizeBytes = statSync(pdfPath).size;
  } catch {}

  const result: DocumentResult = {
    filename,
    fileSizeBytes,
    pageCount: 0,
    totalChars: 0,
    charsPerPage: 0,
    status: 'failed',
    ocrStatus: 'skip',
    embeddingStatus: 'skip',
    citationStatus: 'skip',
    summaryStatus: 'skip',
    ocrText: '',
    citation: '',
    summary: '',
    embedding: [],
    timings: {
      pdfToImageMs: 0,
      ocrMs: 0,
      ocrPerPageMs: 0,
      embeddingMs: 0,
      citationMs: 0,
      summaryMs: 0,
      totalMs: 0,
    },
  };

  try {
    // Step 1: PDF to images
    const imagesDir = join(OUTPUT_DIR, 'images');
    mkdirSync(imagesDir, { recursive: true });
    const { paths: imagePaths, timeMs: pdfToImageMs } = await pdfToImages(pdfPath, imagesDir);
    result.pageCount = imagePaths.length;
    result.timings.pdfToImageMs = pdfToImageMs;

    if (imagePaths.length === 0) {
      result.error = 'No pages converted';
      return result;
    }

    // Step 2: OCR
    try {
      const { ocrText, timeMs: ocrMs } = await processOcrParallel(imagePaths);
      result.ocrText = ocrText;
      result.totalChars = ocrText.length;
      result.charsPerPage = Math.round(ocrText.length / imagePaths.length);
      result.timings.ocrMs = ocrMs;
      result.timings.ocrPerPageMs = ocrMs / imagePaths.length;
      result.ocrStatus = ocrText.length > 100 ? 'pass' : 'fail';
    } catch (e) {
      result.ocrStatus = 'fail';
      result.error = `OCR failed: ${e instanceof Error ? e.message : e}`;
    }

    // Step 3: Embedding (only if OCR succeeded)
    if (result.ocrStatus === 'pass') {
      try {
        const { embedding, timeMs: embeddingMs } = await generateEmbedding(result.ocrText);
        result.embedding = embedding;
        result.timings.embeddingMs = embeddingMs;
        result.embeddingStatus = embedding.length === 768 ? 'pass' : 'fail';
      } catch (e) {
        result.embeddingStatus = 'fail';
        result.error = `Embedding failed: ${e instanceof Error ? e.message : e}`;
      }
    }

    // Step 4: Citation (only if OCR succeeded)
    if (result.ocrStatus === 'pass') {
      try {
        const { citation, timeMs: citationMs } = await generateCitation(result.ocrText, filename);
        result.citation = citation;
        result.timings.citationMs = citationMs;
        result.citationStatus = citation.length > 10 ? 'pass' : 'fail';
      } catch (e) {
        result.citationStatus = 'fail';
        result.error = `Citation failed: ${e instanceof Error ? e.message : e}`;
      }
    }

    // Step 5: Summary (only if OCR succeeded)
    if (result.ocrStatus === 'pass') {
      try {
        const { summary, timeMs: summaryMs } = await generateSummary(result.ocrText);
        result.summary = summary;
        result.timings.summaryMs = summaryMs;
        result.summaryStatus = summary.length > 20 ? 'pass' : 'fail';
      } catch (e) {
        result.summaryStatus = 'fail';
        result.error = `Summary failed: ${e instanceof Error ? e.message : e}`;
      }
    }

    // Cleanup images
    for (const imgPath of imagePaths) {
      try { unlinkSync(imgPath); } catch {}
    }

    // Overall status - require OCR, embedding, and citation (summary is optional for pass)
    result.status = (
      result.ocrStatus === 'pass' &&
      result.embeddingStatus === 'pass' &&
      result.citationStatus === 'pass'
    ) ? 'success' : 'failed';

  } catch (e) {
    result.error = e instanceof Error ? e.message : String(e);
  }

  result.timings.totalMs = performance.now() - docStart;
  return result;
}

// ============================================================
// Iteration Runner
// ============================================================

async function runIteration(
  pdfFiles: string[],
  iterationNum: number,
  direction: 'forward' | 'backward'
): Promise<IterationResult> {
  const iterStart = performance.now();
  const results: DocumentResult[] = [];

  const orderedFiles = direction === 'backward' ? [...pdfFiles].reverse() : pdfFiles;

  console.log(chalk.cyan.bold(`\n${'─'.repeat(70)}`));
  console.log(chalk.cyan.bold(`  Iteration ${iterationNum} (${direction})`));
  console.log(chalk.cyan.bold(`${'─'.repeat(70)}`));

  for (let i = 0; i < orderedFiles.length; i++) {
    const pdfPath = orderedFiles[i];
    const filename = basename(pdfPath);

    console.log(chalk.white(`\n  [${i + 1}/${orderedFiles.length}] ${filename}`));

    const result = await processDocument(pdfPath);
    results.push(result);

    // Inline status
    const ocrIcon = result.ocrStatus === 'pass' ? chalk.green('OK') : chalk.red('FAIL');
    const embIcon = result.embeddingStatus === 'pass' ? chalk.green('OK') : chalk.red('FAIL');
    const citIcon = result.citationStatus === 'pass' ? chalk.green('OK') : chalk.red('FAIL');
    const sumIcon = result.summaryStatus === 'pass' ? chalk.green('OK') : chalk.yellow('--');

    console.log(`    Pages: ${result.pageCount} | Chars: ${result.totalChars.toLocaleString()} (${result.charsPerPage}/pg)`);
    console.log(`    OCR: ${ocrIcon} | Embed: ${embIcon} | Citation: ${citIcon} | Summary: ${sumIcon}`);

    if (result.citationStatus === 'pass') {
      console.log(chalk.dim(`    Citation: ${result.citation.slice(0, 65)}...`));
    }
    if (result.summaryStatus === 'pass') {
      console.log(chalk.dim(`    Summary: ${result.summary.slice(0, 65)}...`));
    }
    if (result.error) {
      console.log(chalk.red(`    Error: ${result.error}`));
    }

    const ocrPerPage = result.pageCount > 0 ? (result.timings.ocrMs / result.pageCount / 1000).toFixed(2) : '0';
    console.log(chalk.dim(`    Time: ${(result.timings.totalMs / 1000).toFixed(1)}s (OCR: ${ocrPerPage}s/pg)`));
  }

  // Calculate metrics
  const totalPages = results.reduce((sum, r) => sum + r.pageCount, 0);
  const totalChars = results.reduce((sum, r) => sum + r.totalChars, 0);
  const totalTimeMs = performance.now() - iterStart;

  const validOcrResults = results.filter(r => r.ocrStatus === 'pass');
  const avgOcrPerPageMs = validOcrResults.length > 0
    ? validOcrResults.reduce((sum, r) => sum + r.timings.ocrPerPageMs, 0) / validOcrResults.length
    : 0;
  const avgOcrPerDocMs = validOcrResults.length > 0
    ? validOcrResults.reduce((sum, r) => sum + r.timings.ocrMs, 0) / validOcrResults.length
    : 0;
  const avgEmbeddingMs = validOcrResults.length > 0
    ? validOcrResults.reduce((sum, r) => sum + r.timings.embeddingMs, 0) / validOcrResults.length
    : 0;
  const avgCitationMs = validOcrResults.length > 0
    ? validOcrResults.reduce((sum, r) => sum + r.timings.citationMs, 0) / validOcrResults.length
    : 0;
  const avgSummaryMs = validOcrResults.length > 0
    ? validOcrResults.reduce((sum, r) => sum + r.timings.summaryMs, 0) / validOcrResults.length
    : 0;
  const avgTotalPerDocMs = results.length > 0
    ? results.reduce((sum, r) => sum + r.timings.totalMs, 0) / results.length
    : 0;

  const metrics = {
    total: results.length,
    passed: results.filter(r => r.status === 'success').length,
    failed: results.filter(r => r.status === 'failed').length,
    ocrPassed: results.filter(r => r.ocrStatus === 'pass').length,
    embeddingPassed: results.filter(r => r.embeddingStatus === 'pass').length,
    citationPassed: results.filter(r => r.citationStatus === 'pass').length,
    summaryPassed: results.filter(r => r.summaryStatus === 'pass').length,
    totalPages,
    totalChars,
    avgCharsPerPage: totalPages > 0 ? Math.round(totalChars / totalPages) : 0,
    avgOcrPerPageMs,
    avgOcrPerDocMs,
    avgEmbeddingMs,
    avgCitationMs,
    avgSummaryMs,
    avgTotalPerDocMs,
    totalTimeMs,
    docsPerMinute: (results.length / (totalTimeMs / 60000)),
    pagesPerMinute: (totalPages / (totalTimeMs / 60000)),
  };

  return { iteration: iterationNum, direction, documents: results, metrics };
}

// ============================================================
// Main
// ============================================================

async function main() {
  const args = process.argv.slice(2);

  // Parse arguments
  const iterIdx = args.indexOf('--iterations');
  const iterations = iterIdx !== -1 ? parseInt(args[iterIdx + 1]) || 1 : 1;
  const forwardBackward = args.includes('--forward-backward');
  const reverseOnly = args.includes('--reverse');
  const pdfIdx = args.indexOf('--pdf');
  const specificPdf = pdfIdx !== -1 ? args[pdfIdx + 1] : null;

  console.log(chalk.cyan.bold('\n' + '═'.repeat(70)));
  console.log(chalk.cyan.bold('     Comprehensive Benchmark - OCR + Embed + Citation + Summary'));
  console.log(chalk.cyan.bold('═'.repeat(70)));
  console.log(chalk.dim(`  Iterations: ${iterations}${forwardBackward ? ' (forward + backward)' : reverseOnly ? ' (backward)' : ' (forward)'}`));

  // Check services
  console.log(chalk.white.bold('\n  Service Check'));
  console.log(chalk.dim('  ' + '─'.repeat(50)));

  const ocrEndpoints = await getAvailableOcrEndpoints();
  const embStatus = await checkService(EMBEDDINGS_LB_URL);
  const infUrl = await findInferenceEndpoint();

  console.log(`  OCR: ${ocrEndpoints.length > 0 ? chalk.green(`${ocrEndpoints.length} endpoint(s)`) : chalk.red('NONE')}`);
  console.log(`  Embeddings: ${embStatus.healthy ? chalk.green('OK') : chalk.red('DOWN')}`);
  console.log(`  Inference: ${infUrl ? chalk.green('OK') : chalk.red('DOWN')}`);

  if (ocrEndpoints.length === 0 || !embStatus.healthy || !infUrl) {
    console.log(chalk.red('\n  Required services not available. Start the cluster.'));
    process.exit(1);
  }

  // Get PDF files
  let pdfFiles: string[] = [];
  if (specificPdf) {
    if (!existsSync(specificPdf)) {
      console.log(chalk.red(`\n  PDF not found: ${specificPdf}`));
      process.exit(1);
    }
    pdfFiles = [specificPdf];
  } else {
    if (!existsSync(FILES_DIR)) {
      console.log(chalk.red(`\n  Files directory not found: ${FILES_DIR}`));
      process.exit(1);
    }
    pdfFiles = readdirSync(FILES_DIR)
      .filter(f => f.endsWith('.pdf'))
      .sort()
      .map(f => join(FILES_DIR, f));
  }

  console.log(chalk.white(`\n  PDFs to process: ${pdfFiles.length}`));

  mkdirSync(OUTPUT_DIR, { recursive: true });

  // Run iterations
  const allResults: IterationResult[] = [];
  const overallStart = performance.now();

  for (let i = 1; i <= iterations; i++) {
    if (forwardBackward) {
      // Forward pass
      const fwdResult = await runIteration(pdfFiles, i * 2 - 1, 'forward');
      allResults.push(fwdResult);
      // Backward pass
      const bwdResult = await runIteration(pdfFiles, i * 2, 'backward');
      allResults.push(bwdResult);
    } else if (reverseOnly) {
      const result = await runIteration(pdfFiles, i, 'backward');
      allResults.push(result);
    } else {
      const result = await runIteration(pdfFiles, i, 'forward');
      allResults.push(result);
    }
  }

  const overallTime = performance.now() - overallStart;

  // ================================================================
  // COMPREHENSIVE BENCHMARK REPORT
  // ================================================================

  console.log(chalk.cyan.bold('\n' + '═'.repeat(70)));
  console.log(chalk.cyan.bold('                    BENCHMARK REPORT'));
  console.log(chalk.cyan.bold('═'.repeat(70)));

  // Per-iteration summary
  console.log(chalk.white.bold('\n  Per-Iteration Results:'));
  console.log(chalk.dim('  ' + '─'.repeat(60)));
  console.log(chalk.dim('  Iter  Dir       Pass    Pages   Chars      OCR/pg   Time'));
  console.log(chalk.dim('  ' + '─'.repeat(60)));

  for (const iter of allResults) {
    const passRate = ((iter.metrics.passed / iter.metrics.total) * 100).toFixed(0);
    const statusColor = iter.metrics.failed === 0 ? chalk.green : chalk.yellow;
    const dir = iter.direction === 'forward' ? '→ fwd' : '← bwd';
    console.log(
      `  ${String(iter.iteration).padStart(2)}    ${dir}    ${statusColor(`${iter.metrics.passed}/${iter.metrics.total}`.padEnd(6))}  ` +
      `${String(iter.metrics.totalPages).padStart(5)}   ${iter.metrics.totalChars.toLocaleString().padStart(8)}   ` +
      `${(iter.metrics.avgOcrPerPageMs / 1000).toFixed(2)}s    ${(iter.metrics.totalTimeMs / 1000).toFixed(1)}s`
    );
  }

  // Aggregate stats
  const totalDocs = allResults.reduce((sum, r) => sum + r.metrics.total, 0);
  const totalPages = allResults.reduce((sum, r) => sum + r.metrics.totalPages, 0);
  const totalChars = allResults.reduce((sum, r) => sum + r.metrics.totalChars, 0);
  const totalPassed = allResults.reduce((sum, r) => sum + r.metrics.passed, 0);
  const totalFailed = allResults.reduce((sum, r) => sum + r.metrics.failed, 0);

  const avgOcrPerPageMs = allResults.reduce((sum, r) => sum + r.metrics.avgOcrPerPageMs, 0) / allResults.length;
  const avgOcrPerDocMs = allResults.reduce((sum, r) => sum + r.metrics.avgOcrPerDocMs, 0) / allResults.length;
  const avgEmbeddingMs = allResults.reduce((sum, r) => sum + r.metrics.avgEmbeddingMs, 0) / allResults.length;
  const avgCitationMs = allResults.reduce((sum, r) => sum + r.metrics.avgCitationMs, 0) / allResults.length;
  const avgSummaryMs = allResults.reduce((sum, r) => sum + r.metrics.avgSummaryMs, 0) / allResults.length;
  const avgTotalPerDocMs = allResults.reduce((sum, r) => sum + r.metrics.avgTotalPerDocMs, 0) / allResults.length;
  const docsPerMinute = (totalDocs / (overallTime / 60000));
  const pagesPerMinute = (totalPages / (overallTime / 60000));

  console.log(chalk.white.bold('\n  Aggregate Statistics:'));
  console.log(chalk.dim('  ' + '─'.repeat(50)));
  console.log(`  Documents processed: ${totalDocs}`);
  console.log(`  Total pages: ${totalPages}`);
  console.log(`  Total characters: ${totalChars.toLocaleString()}`);
  console.log(`  Pass rate: ${((totalPassed / totalDocs) * 100).toFixed(1)}%`);
  console.log(`  Total time: ${(overallTime / 1000).toFixed(1)}s`);

  console.log(chalk.white.bold('\n  Performance Metrics:'));
  console.log(chalk.dim('  ' + '─'.repeat(50)));
  console.log(`  Avg OCR per page:     ${(avgOcrPerPageMs / 1000).toFixed(2)}s`);
  console.log(`  Avg OCR per document: ${(avgOcrPerDocMs / 1000).toFixed(2)}s`);
  console.log(`  Avg embedding:        ${avgEmbeddingMs.toFixed(0)}ms`);
  console.log(`  Avg citation:         ${(avgCitationMs / 1000).toFixed(2)}s`);
  console.log(`  Avg summary:          ${(avgSummaryMs / 1000).toFixed(2)}s`);
  console.log(`  Avg total per doc:    ${(avgTotalPerDocMs / 1000).toFixed(2)}s`);

  console.log(chalk.white.bold('\n  Throughput:'));
  console.log(chalk.dim('  ' + '─'.repeat(50)));
  console.log(`  Documents per minute: ${docsPerMinute.toFixed(2)}`);
  console.log(`  Pages per minute:     ${pagesPerMinute.toFixed(2)}`);

  // Projections
  const docs100Minutes = 100 / docsPerMinute;
  const docs1000Minutes = 1000 / docsPerMinute;

  console.log(chalk.white.bold('\n  Projections:'));
  console.log(chalk.dim('  ' + '─'.repeat(50)));
  console.log(`  100 documents:  ${docs100Minutes.toFixed(1)} min (${(docs100Minutes / 60).toFixed(2)} hrs)`);
  console.log(`  1000 documents: ${docs1000Minutes.toFixed(1)} min (${(docs1000Minutes / 60).toFixed(2)} hrs)`);

  // Document-level breakdown
  console.log(chalk.white.bold('\n  Document-Level Results (first iteration):'));
  console.log(chalk.dim('  ' + '─'.repeat(60)));

  const firstIter = allResults[0];
  for (const doc of firstIter.documents) {
    const icon = doc.status === 'success' ? chalk.green('PASS') : chalk.red('FAIL');
    console.log(`  ${icon} ${doc.filename.slice(0, 25).padEnd(25)} ${String(doc.pageCount).padStart(3)}pg ${doc.totalChars.toLocaleString().padStart(7)}ch ${(doc.timings.totalMs / 1000).toFixed(1).padStart(5)}s`);
    if (doc.citationStatus === 'pass') {
      console.log(chalk.dim(`       └─ ${doc.citation.slice(0, 55)}...`));
    }
  }

  // Overall verdict
  console.log(chalk.cyan.bold('\n' + '═'.repeat(70)));
  const allPassed = totalFailed === 0;
  if (allPassed) {
    console.log(chalk.green.bold('  BENCHMARK PASSED - All documents processed successfully'));
  } else {
    console.log(chalk.yellow.bold(`  BENCHMARK COMPLETE - ${totalPassed}/${totalDocs} documents passed (${((totalPassed/totalDocs)*100).toFixed(1)}%)`));
  }
  console.log(chalk.cyan.bold('═'.repeat(70) + '\n'));

  // Save comprehensive report
  const report: BenchmarkReport = {
    timestamp: new Date().toISOString(),
    config: {
      iterations,
      forwardBackward,
      pdfCount: pdfFiles.length,
      filesDir: FILES_DIR,
    },
    services: {
      ocrEndpoints: ocrEndpoints.length,
      embeddingsOk: embStatus.healthy,
      inferenceOk: !!infUrl,
    },
    iterations: allResults,
    aggregate: {
      totalDocs,
      totalPages,
      totalChars,
      passRate: (totalPassed / totalDocs) * 100,
      avgOcrPerPageMs,
      avgOcrPerDocMs,
      avgEmbeddingMs,
      avgCitationMs,
      avgSummaryMs,
      avgTotalPerDocMs,
      docsPerMinute,
      pagesPerMinute,
    },
    projections: {
      docs100: { estimatedMinutes: docs100Minutes, estimatedHours: docs100Minutes / 60 },
      docs1000: { estimatedMinutes: docs1000Minutes, estimatedHours: docs1000Minutes / 60 },
    },
    verdict: allPassed ? 'PASSED' : 'FAILED',
  };

  const resultsPath = join(OUTPUT_DIR, `benchmark-${Date.now()}.json`);
  writeFileSync(resultsPath, JSON.stringify(report, null, 2));

  console.log(chalk.dim(`  Report saved: ${resultsPath}\n`));

  process.exit(allPassed ? 0 : 1);
}

main().catch(error => {
  console.log(chalk.red(`\nFatal error: ${error.message}`));
  console.error(error);
  process.exit(1);
});
