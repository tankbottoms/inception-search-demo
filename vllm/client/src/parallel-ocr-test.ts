/**
 * Parallel OCR Test - Verify true concurrent execution across spark-1 and spark-2
 *
 * This test:
 *   1. Generates test images in memory with distinct text
 *   2. Sends them to multiple OCR endpoints TRULY in parallel
 *   3. Monitors GPU usage on both nodes during execution
 *   4. Reports timing and throughput metrics
 *
 * Usage:
 *   bun run src/parallel-ocr-test.ts              # Default 4 images
 *   bun run src/parallel-ocr-test.ts --count 8    # Custom count
 *   bun run src/parallel-ocr-test.ts --sequential # Compare with sequential
 */

import axios from 'axios';
import chalk from 'chalk';
import { execSync } from 'child_process';
import sharp from 'sharp';

// ============================================================
// Configuration
// ============================================================

const OCR_ENDPOINTS = [
  { url: 'http://localhost:8003', host: 'spark-1', ip: 'localhost' },
  { url: 'http://192.168.1.63:8003', host: 'spark-2', ip: '192.168.1.63' },
];

const EMBEDDING_ENDPOINTS = [
  { url: 'http://localhost:8001', host: 'spark-1:8001' },
  { url: 'http://localhost:8002', host: 'spark-1:8002' },
  { url: 'http://192.168.1.63:8001', host: 'spark-2:8001' },
  { url: 'http://192.168.1.63:8002', host: 'spark-2:8002' },
];

interface OcrEndpoint {
  url: string;
  host: string;
  ip: string;
  healthy: boolean;
  modelId: string | null;
}

interface GpuStats {
  host: string;
  utilization: number;
  memoryUsed: string;
  timestamp: number;
}

// ============================================================
// GPU Monitoring
// ============================================================

async function getGpuStats(host: string, ip: string): Promise<GpuStats | null> {
  try {
    const cmd = ip === 'localhost'
      ? 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits'
      : `ssh rooot@${ip} "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" 2>/dev/null`;

    const output = execSync(cmd, { encoding: 'utf-8', timeout: 5000 });
    const [util, mem] = output.trim().split(',').map(s => s.trim());

    return {
      host,
      utilization: parseFloat(util) || 0,
      memoryUsed: mem || 'N/A',
      timestamp: Date.now(),
    };
  } catch {
    return null;
  }
}

class GpuMonitor {
  private samples: GpuStats[] = [];
  private interval: NodeJS.Timer | null = null;
  private hosts: { host: string; ip: string }[];

  constructor(hosts: { host: string; ip: string }[]) {
    this.hosts = hosts;
  }

  start(intervalMs: number = 500) {
    this.samples = [];
    this.interval = setInterval(async () => {
      for (const { host, ip } of this.hosts) {
        const stats = await getGpuStats(host, ip);
        if (stats) this.samples.push(stats);
      }
    }, intervalMs);
  }

  stop(): { samples: GpuStats[]; summary: Record<string, { peak: number; avg: number; samples: number }> } {
    if (this.interval) clearInterval(this.interval);

    const summary: Record<string, { peak: number; avg: number; samples: number }> = {};

    for (const { host } of this.hosts) {
      const hostSamples = this.samples.filter(s => s.host === host);
      if (hostSamples.length > 0) {
        summary[host] = {
          peak: Math.max(...hostSamples.map(s => s.utilization)),
          avg: hostSamples.reduce((a, b) => a + b.utilization, 0) / hostSamples.length,
          samples: hostSamples.length,
        };
      }
    }

    return { samples: this.samples, summary };
  }
}

// ============================================================
// Image Generation
// ============================================================

async function generateTestImage(text: string, width: number = 800, height: number = 600): Promise<Buffer> {
  // Create SVG with text
  const escapedText = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const lines = escapedText.split('\n');
  const lineHeight = 24;
  const startY = 50;

  const textElements = lines.map((line, i) =>
    `<text x="40" y="${startY + i * lineHeight}" font-family="monospace" font-size="16" fill="black">${line}</text>`
  ).join('\n');

  const svg = `
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      ${textElements}
    </svg>
  `;

  return sharp(Buffer.from(svg)).png().toBuffer();
}

function generateTestText(pageNum: number): string {
  const paragraphs = [
    `TEST DOCUMENT - PAGE ${pageNum}`,
    `Generated at: ${new Date().toISOString()}`,
    '',
    'LEGAL DOCUMENT SIMULATION',
    '=' .repeat(40),
    '',
    `This is paragraph ${pageNum}.1 of the test document.`,
    'It contains sample text that should be extracted by OCR.',
    '',
    `This is paragraph ${pageNum}.2 with some numbers: 12345, 67890.`,
    'And some special characters: @#$%^&*().',
    '',
    `This is paragraph ${pageNum}.3 with legal-style text:`,
    `  - Case No. 2024-CV-${String(pageNum).padStart(4, '0')}`,
    `  - Filed: December 28, 2025`,
    `  - Plaintiff v. Defendant`,
    '',
    `END OF PAGE ${pageNum}`,
  ];
  return paragraphs.join('\n');
}

// ============================================================
// OCR Service
// ============================================================

async function checkOcrEndpoint(endpoint: { url: string; host: string; ip: string }): Promise<OcrEndpoint> {
  try {
    await axios.get(`${endpoint.url}/health`, { timeout: 5000 });
    const models = await axios.get(`${endpoint.url}/v1/models`, { timeout: 5000 });
    return {
      ...endpoint,
      healthy: true,
      modelId: models.data.data?.[0]?.id || null,
    };
  } catch {
    return { ...endpoint, healthy: false, modelId: null };
  }
}

async function runOcr(imageBase64: string, endpoint: OcrEndpoint): Promise<{ text: string; timeMs: number }> {
  const start = performance.now();

  const response = await axios.post(
    `${endpoint.url}/v1/chat/completions`,
    {
      model: endpoint.modelId,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'Extract ALL text from this image. Output only the text.' },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
        ],
      }],
      max_tokens: 2048,
      temperature: 0.1,
    },
    { timeout: 120000 }
  );

  return {
    text: response.data.choices[0].message.content || '',
    timeMs: performance.now() - start,
  };
}

// ============================================================
// Parallel Processing
// ============================================================

interface OcrTask {
  pageNum: number;
  imageBase64: string;
  endpoint: OcrEndpoint;
}

interface OcrResult {
  pageNum: number;
  endpoint: string;
  text: string;
  timeMs: number;
  startTime: number;
  endTime: number;
}

async function processParallel(
  images: { pageNum: number; base64: string }[],
  endpoints: OcrEndpoint[],
  concurrency: number
): Promise<OcrResult[]> {
  const results: OcrResult[] = [];
  const queue: OcrTask[] = [];

  // Build task queue - distribute across endpoints
  for (let i = 0; i < images.length; i++) {
    const endpoint = endpoints[i % endpoints.length];
    queue.push({
      pageNum: images[i].pageNum,
      imageBase64: images[i].base64,
      endpoint,
    });
  }

  // Process with controlled concurrency
  const inFlight: Promise<void>[] = [];
  let queueIndex = 0;

  const processTask = async (task: OcrTask): Promise<void> => {
    const startTime = Date.now();
    console.log(chalk.white(`    [START] Page ${task.pageNum} → ${task.endpoint.host}`));

    try {
      const result = await runOcr(task.imageBase64, task.endpoint);
      const endTime = Date.now();

      results.push({
        pageNum: task.pageNum,
        endpoint: task.endpoint.host,
        text: result.text,
        timeMs: result.timeMs,
        startTime,
        endTime,
      });

      console.log(chalk.green(`    [DONE]  Page ${task.pageNum} → ${task.endpoint.host} (${result.timeMs.toFixed(0)}ms)`));
    } catch (error) {
      console.log(chalk.red(`    [ERROR] Page ${task.pageNum} → ${task.endpoint.host}: ${error}`));
    }
  };

  // Launch all tasks up to concurrency limit
  while (queueIndex < queue.length || inFlight.length > 0) {
    // Fill up to concurrency limit
    while (inFlight.length < concurrency && queueIndex < queue.length) {
      const task = queue[queueIndex++];
      const promise = processTask(task).then(() => {
        const idx = inFlight.indexOf(promise);
        if (idx > -1) inFlight.splice(idx, 1);
      });
      inFlight.push(promise);
    }

    // Wait for at least one to complete if at capacity
    if (inFlight.length >= concurrency || queueIndex >= queue.length) {
      await Promise.race(inFlight);
    }
  }

  // Wait for all remaining
  await Promise.all(inFlight);

  return results.sort((a, b) => a.pageNum - b.pageNum);
}

async function processSequential(
  images: { pageNum: number; base64: string }[],
  endpoints: OcrEndpoint[]
): Promise<OcrResult[]> {
  const results: OcrResult[] = [];

  for (let i = 0; i < images.length; i++) {
    const endpoint = endpoints[i % endpoints.length];
    const startTime = Date.now();

    console.log(chalk.white(`    [START] Page ${images[i].pageNum} → ${endpoint.host}`));

    try {
      const result = await runOcr(images[i].base64, endpoint);
      const endTime = Date.now();

      results.push({
        pageNum: images[i].pageNum,
        endpoint: endpoint.host,
        text: result.text,
        timeMs: result.timeMs,
        startTime,
        endTime,
      });

      console.log(chalk.green(`    [DONE]  Page ${images[i].pageNum} → ${endpoint.host} (${result.timeMs.toFixed(0)}ms)`));
    } catch (error) {
      console.log(chalk.red(`    [ERROR] Page ${images[i].pageNum}: ${error}`));
    }
  }

  return results;
}

// ============================================================
// Main
// ============================================================

async function main() {
  const args = process.argv.slice(2);
  const countIdx = args.indexOf('--count');
  const imageCount = countIdx !== -1 ? parseInt(args[countIdx + 1]) : 4;
  const sequential = args.includes('--sequential');

  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('         Parallel OCR Test - True Concurrent Execution'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

  // Check OCR endpoints
  console.log(chalk.white.bold('Checking OCR Endpoints'));
  console.log(chalk.dim('-'.repeat(50)));

  const endpoints: OcrEndpoint[] = [];
  for (const ep of OCR_ENDPOINTS) {
    const status = await checkOcrEndpoint(ep);
    console.log(status.healthy
      ? chalk.green(`  [OK] ${status.host} - ${status.modelId}`)
      : chalk.dim(`  [--] ${status.host}`));
    if (status.healthy) endpoints.push(status);
  }

  if (endpoints.length === 0) {
    console.log(chalk.red('\nNo OCR endpoints available!'));
    process.exit(1);
  }

  console.log(chalk.cyan(`\n  Available endpoints: ${endpoints.length}`));
  console.log(chalk.cyan(`  Mode: ${sequential ? 'Sequential' : 'Parallel'}`));
  console.log(chalk.cyan(`  Images to process: ${imageCount}`));

  // Generate test images
  console.log(chalk.white.bold('\nGenerating Test Images'));
  console.log(chalk.dim('-'.repeat(50)));

  const images: { pageNum: number; base64: string }[] = [];
  for (let i = 1; i <= imageCount; i++) {
    const text = generateTestText(i);
    const buffer = await generateTestImage(text);
    images.push({ pageNum: i, base64: buffer.toString('base64') });
    console.log(chalk.white(`  Generated image ${i}/${imageCount}`));
  }

  // Start GPU monitoring
  console.log(chalk.white.bold('\nStarting GPU Monitor'));
  console.log(chalk.dim('-'.repeat(50)));

  const gpuMonitor = new GpuMonitor(endpoints.map(e => ({ host: e.host, ip: e.ip })));
  gpuMonitor.start(500);
  console.log(chalk.green('  GPU monitoring active'));

  // Process images
  console.log(chalk.white.bold('\nProcessing Images'));
  console.log(chalk.dim('-'.repeat(50)));

  const startTime = performance.now();

  const results = sequential
    ? await processSequential(images, endpoints)
    : await processParallel(images, endpoints, endpoints.length * 2); // 2 concurrent per endpoint

  const totalTime = performance.now() - startTime;

  // Stop GPU monitoring
  const { summary: gpuSummary } = gpuMonitor.stop();

  // Analyze timeline
  console.log(chalk.white.bold('\nExecution Timeline'));
  console.log(chalk.dim('-'.repeat(50)));

  const minStart = Math.min(...results.map(r => r.startTime));
  for (const r of results) {
    const relStart = r.startTime - minStart;
    const relEnd = r.endTime - minStart;
    const bar = ' '.repeat(Math.floor(relStart / 100)) + '█'.repeat(Math.ceil((relEnd - relStart) / 100));
    console.log(chalk.white(`  Page ${r.pageNum} [${r.endpoint}]: ${bar} ${r.timeMs.toFixed(0)}ms`));
  }

  // Calculate overlap (true parallelism indicator)
  let overlapTime = 0;
  for (let i = 0; i < results.length; i++) {
    for (let j = i + 1; j < results.length; j++) {
      const start = Math.max(results[i].startTime, results[j].startTime);
      const end = Math.min(results[i].endTime, results[j].endTime);
      if (end > start) overlapTime += (end - start);
    }
  }

  // Summary
  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('                         Results'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

  console.log(chalk.white.bold('Timing:'));
  console.log(chalk.white(`  Total time:     ${(totalTime / 1000).toFixed(2)}s`));
  console.log(chalk.white(`  Images:         ${results.length}`));
  console.log(chalk.white(`  Throughput:     ${(results.length / (totalTime / 1000)).toFixed(2)} images/sec`));
  console.log(chalk.white(`  Avg per image:  ${(totalTime / results.length).toFixed(0)}ms`));
  console.log(chalk.white(`  Overlap time:   ${(overlapTime / 1000).toFixed(2)}s (${((overlapTime / totalTime) * 100).toFixed(1)}% parallel)`));

  console.log(chalk.white.bold('\nGPU Utilization:'));
  for (const [host, stats] of Object.entries(gpuSummary)) {
    console.log(chalk.white(`  ${host}:`));
    console.log(chalk.white(`    Peak: ${stats.peak.toFixed(1)}%`));
    console.log(chalk.white(`    Avg:  ${stats.avg.toFixed(1)}%`));
    console.log(chalk.white(`    Samples: ${stats.samples}`));
  }

  console.log(chalk.white.bold('\nEndpoint Distribution:'));
  for (const ep of endpoints) {
    const count = results.filter(r => r.endpoint === ep.host).length;
    console.log(chalk.white(`  ${ep.host}: ${count} images`));
  }

  // Verify text extraction
  console.log(chalk.white.bold('\nText Extraction Verification:'));
  let verified = 0;
  for (const r of results) {
    const expectedMarker = `PAGE ${r.pageNum}`;
    const found = r.text.includes(expectedMarker) || r.text.includes(`Page ${r.pageNum}`);
    if (found) {
      verified++;
      console.log(chalk.green(`  Page ${r.pageNum}: OK (found marker)`));
    } else {
      console.log(chalk.red(`  Page ${r.pageNum}: FAILED (marker not found)`));
    }
  }

  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  const passRate = (verified / results.length) * 100;
  if (passRate === 100 && overlapTime > totalTime * 0.3) {
    console.log(chalk.green.bold('  TEST PASSED - True parallel execution confirmed'));
  } else if (passRate === 100) {
    console.log(chalk.yellow.bold('  TEST PARTIAL - OCR works but limited parallelism'));
  } else {
    console.log(chalk.red.bold('  TEST FAILED - OCR extraction errors'));
  }
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));
}

main().catch(console.error);
