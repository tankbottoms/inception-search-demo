/**
 * Inception ONNX - CLI Tool
 *
 * Commands:
 *   --check     Validate and download/convert models
 *   --benchmark Run performance benchmarks (CPU vs GPU)
 */

import { Command } from 'commander';
import chalk from 'chalk';
import {
  loadRegistry,
  checkModelAvailable,
  loadModel,
  detectHardware,
  getHardwareStatus,
  initEmbeddingService,
  generateQueryEmbedding,
  generateTextEmbedding,
  generateBatchEmbeddings,
  settings,
  Timer,
} from './services';
import type { ModelEntry } from './types';

const program = new Command();

program
  .name('inception-cli')
  .description('Inception ONNX CLI tool')
  .version('2.0.0');

program
  .option('--check', 'Validate and download/convert models')
  .option('--benchmark', 'Run performance benchmarks')
  .option('--provider <provider>', 'Force execution provider (cpu|cuda)', '')
  .option('--iterations <n>', 'Number of benchmark iterations', '10');

program.parse();

const options = program.opts();

// ============================================================
// Model Check Command
// ============================================================

async function checkModels(): Promise<void> {
  console.log(chalk.blue('\n━━━ Inception ONNX Model Check ━━━\n'));

  const hw = detectHardware();
  console.log(chalk.gray(`Hardware: ${hw.deviceName}`));
  console.log(chalk.gray(`Provider: ${hw.provider}`));
  if (hw.cudaVersion) {
    console.log(chalk.gray(`CUDA: ${hw.cudaVersion}`));
  }
  console.log('');

  const registry = loadRegistry();
  console.log(chalk.gray(`Registry: ${registry.cache_dir}/registry.json`));
  console.log(chalk.gray(`Version: ${registry.version}`));
  console.log('');

  let hasIssues = false;

  for (const model of registry.models) {
    console.log(chalk.white.bold(`📦 ${model.id}`));
    console.log(chalk.gray(`   Name: ${model.name}`));
    console.log(chalk.gray(`   Type: ${model.type}`));
    console.log(chalk.gray(`   Enabled: ${model.enabled}`));

    if (!model.enabled) {
      console.log(chalk.yellow(`   Status: disabled`));
      console.log('');
      continue;
    }

    if (model.status === 'planned') {
      console.log(chalk.yellow(`   Status: planned (not yet available)`));
      console.log('');
      continue;
    }

    // Check availability
    const availability = await checkModelAvailable(model.id);

    if (availability.available) {
      if (availability.location === 'cache') {
        console.log(chalk.green(`   ✓ Available in local cache`));
        console.log(chalk.gray(`     Path: ${availability.path}`));
      } else if (availability.location === 'huggingface') {
        console.log(chalk.blue(`   ↓ Available on HuggingFace (will download on first use)`));
      } else if (availability.location === 'converter') {
        console.log(chalk.yellow(`   ⚙ Requires conversion (converter service available)`));
      }
    } else {
      hasIssues = true;
      console.log(chalk.red(`   ✗ Not available: ${availability.error}`));
    }

    // Show config if available
    if (model.config) {
      const cfg = model.config;
      if (cfg.embedding_dim) {
        console.log(chalk.gray(`   Embedding dim: ${cfg.embedding_dim}`));
      }
      if (cfg.max_tokens) {
        console.log(chalk.gray(`   Max tokens: ${cfg.max_tokens}`));
      }
    }

    console.log('');
  }

  // Summary
  console.log(chalk.blue('━━━ Summary ━━━\n'));

  if (hasIssues) {
    console.log(chalk.yellow('⚠ Some models require attention.'));
    console.log(chalk.gray('  Run the converter service to prepare missing models:'));
    console.log(chalk.gray('  $ docker compose up converter'));
  } else {
    console.log(chalk.green('✓ All enabled models are available or can be downloaded.'));
  }

  console.log('');
}

// ============================================================
// Benchmark Command
// ============================================================

interface BenchmarkResult {
  name: string;
  provider: string;
  iterations: number;
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  std_dev: number;
  throughput: number;
}

function calculateStats(times: number[]): { avg: number; min: number; max: number; stdDev: number } {
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const min = Math.min(...times);
  const max = Math.max(...times);
  const variance = times.reduce((sum, t) => sum + (t - avg) ** 2, 0) / times.length;
  const stdDev = Math.sqrt(variance);
  return { avg, min, max, stdDev };
}

async function runBenchmarks(): Promise<void> {
  const iterations = parseInt(options.iterations, 10) || 10;

  console.log(chalk.blue('\n━━━ Inception ONNX Benchmark ━━━\n'));

  const hw = getHardwareStatus();
  console.log(chalk.white.bold('Hardware Configuration:'));
  console.log(chalk.gray(`  Device: ${hw.deviceName}`));
  console.log(chalk.gray(`  Provider: ${hw.provider}`));
  if (hw.cudaVersion) {
    console.log(chalk.gray(`  CUDA: ${hw.cudaVersion}`));
  }
  if (hw.memoryTotal) {
    console.log(chalk.gray(`  GPU Memory: ${hw.memoryFree}/${hw.memoryTotal} MB free`));
  }
  console.log('');

  // Initialize service
  console.log(chalk.gray('Initializing embedding service...'));
  const initTimer = new Timer('init');
  await initEmbeddingService();
  console.log(chalk.gray(`  Initialization: ${initTimer.elapsed().toFixed(2)}ms`));
  console.log('');

  const results: BenchmarkResult[] = [];

  // Test data
  const shortQuery = 'What is the legal precedent for this case?';
  const longQuery = 'The Supreme Court has established that constitutional rights must be balanced against state interests, considering factors such as public safety, individual liberty, and due process requirements under the Fourteenth Amendment.';

  const shortDocument = 'The court finds in favor of the plaintiff. This ruling is based on established precedent.';
  const longDocument = `
    The United States Supreme Court, in its landmark decision, addressed the fundamental question
    of constitutional interpretation that has been debated since the founding of the republic.
    The majority opinion, written by the Chief Justice, articulated a framework for analyzing
    similar cases in the future. The dissenting justices raised important concerns about the
    potential implications of this ruling on state sovereignty and individual rights.

    The Court examined the historical context of the constitutional provision at issue,
    tracing its origins back to the debates of the Constitutional Convention. The Framers'
    intent, as evidenced by contemporaneous writings and the Federalist Papers, was carefully
    considered in reaching this decision.

    Furthermore, the Court analyzed the precedential value of previous rulings in this area,
    distinguishing the present case from earlier decisions while reaffirming core principles
    that have guided constitutional jurisprudence for over two centuries.

    In conclusion, the Court held that the challenged statute must be examined under strict
    scrutiny, and that the government bears the burden of demonstrating a compelling state
    interest that cannot be achieved through less restrictive means.
  `.trim();

  console.log(chalk.white.bold('Running Benchmarks:'));
  console.log(chalk.gray(`  Iterations per test: ${iterations}`));
  console.log('');

  // Warmup
  console.log(chalk.gray('  Warming up (2 iterations)...'));
  await generateQueryEmbedding(shortQuery);
  await generateQueryEmbedding(shortQuery);

  // Benchmark 1: Short query embedding
  console.log(chalk.white('  1. Short Query Embedding'));
  let times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const timer = new Timer('query');
    await generateQueryEmbedding(shortQuery);
    times.push(timer.elapsed());
  }
  let stats = calculateStats(times);
  results.push({
    name: 'Short Query',
    provider: hw.provider,
    iterations,
    avg_ms: stats.avg,
    min_ms: stats.min,
    max_ms: stats.max,
    std_dev: stats.stdDev,
    throughput: 1000 / stats.avg,
  });
  console.log(chalk.green(`     Avg: ${stats.avg.toFixed(2)}ms (±${stats.stdDev.toFixed(2)}ms)`));

  // Benchmark 2: Long query embedding
  console.log(chalk.white('  2. Long Query Embedding'));
  times = [];
  for (let i = 0; i < iterations; i++) {
    const timer = new Timer('query');
    await generateQueryEmbedding(longQuery);
    times.push(timer.elapsed());
  }
  stats = calculateStats(times);
  results.push({
    name: 'Long Query',
    provider: hw.provider,
    iterations,
    avg_ms: stats.avg,
    min_ms: stats.min,
    max_ms: stats.max,
    std_dev: stats.stdDev,
    throughput: 1000 / stats.avg,
  });
  console.log(chalk.green(`     Avg: ${stats.avg.toFixed(2)}ms (±${stats.stdDev.toFixed(2)}ms)`));

  // Benchmark 3: Short document with chunking
  console.log(chalk.white('  3. Short Document Embedding'));
  times = [];
  for (let i = 0; i < iterations; i++) {
    const timer = new Timer('doc');
    await generateTextEmbedding(1, shortDocument);
    times.push(timer.elapsed());
  }
  stats = calculateStats(times);
  results.push({
    name: 'Short Document',
    provider: hw.provider,
    iterations,
    avg_ms: stats.avg,
    min_ms: stats.min,
    max_ms: stats.max,
    std_dev: stats.stdDev,
    throughput: 1000 / stats.avg,
  });
  console.log(chalk.green(`     Avg: ${stats.avg.toFixed(2)}ms (±${stats.stdDev.toFixed(2)}ms)`));

  // Benchmark 4: Long document with chunking
  console.log(chalk.white('  4. Long Document Embedding'));
  times = [];
  for (let i = 0; i < iterations; i++) {
    const timer = new Timer('doc');
    await generateTextEmbedding(1, longDocument);
    times.push(timer.elapsed());
  }
  stats = calculateStats(times);
  results.push({
    name: 'Long Document',
    provider: hw.provider,
    iterations,
    avg_ms: stats.avg,
    min_ms: stats.min,
    max_ms: stats.max,
    std_dev: stats.stdDev,
    throughput: 1000 / stats.avg,
  });
  console.log(chalk.green(`     Avg: ${stats.avg.toFixed(2)}ms (±${stats.stdDev.toFixed(2)}ms)`));

  // Benchmark 5: Batch processing
  console.log(chalk.white('  5. Batch Processing (5 documents)'));
  const batchDocs = [
    { id: 1, text: shortDocument },
    { id: 2, text: longDocument },
    { id: 3, text: shortDocument + ' Additional context.' },
    { id: 4, text: longDocument.slice(0, longDocument.length / 2) },
    { id: 5, text: shortDocument.repeat(3) },
  ];
  times = [];
  for (let i = 0; i < iterations; i++) {
    const timer = new Timer('batch');
    await generateBatchEmbeddings(batchDocs);
    times.push(timer.elapsed());
  }
  stats = calculateStats(times);
  results.push({
    name: 'Batch (5 docs)',
    provider: hw.provider,
    iterations,
    avg_ms: stats.avg,
    min_ms: stats.min,
    max_ms: stats.max,
    std_dev: stats.stdDev,
    throughput: (5 * 1000) / stats.avg, // docs/sec
  });
  console.log(chalk.green(`     Avg: ${stats.avg.toFixed(2)}ms (±${stats.stdDev.toFixed(2)}ms)`));

  // Results summary
  console.log(chalk.blue('\n━━━ Results Summary ━━━\n'));

  console.log(chalk.white.bold('Performance Table:'));
  console.log(chalk.gray('┌─────────────────────┬──────────┬──────────┬──────────┬────────────┐'));
  console.log(chalk.gray('│ Test                │ Avg (ms) │ Min (ms) │ Max (ms) │ Throughput │'));
  console.log(chalk.gray('├─────────────────────┼──────────┼──────────┼──────────┼────────────┤'));

  for (const r of results) {
    const throughputStr = r.name.includes('Batch')
      ? `${r.throughput.toFixed(1)} doc/s`
      : `${r.throughput.toFixed(1)} req/s`;
    console.log(
      chalk.gray('│ ') +
        r.name.padEnd(19) +
        chalk.gray(' │ ') +
        r.avg_ms.toFixed(2).padStart(8) +
        chalk.gray(' │ ') +
        r.min_ms.toFixed(2).padStart(8) +
        chalk.gray(' │ ') +
        r.max_ms.toFixed(2).padStart(8) +
        chalk.gray(' │ ') +
        throughputStr.padStart(10) +
        chalk.gray(' │')
    );
  }

  console.log(chalk.gray('└─────────────────────┴──────────┴──────────┴──────────┴────────────┘'));
  console.log('');

  console.log(chalk.white.bold('Environment:'));
  console.log(chalk.gray(`  Provider: ${hw.provider.toUpperCase()}`));
  console.log(chalk.gray(`  Device: ${hw.deviceName}`));
  console.log(chalk.gray(`  Max Tokens: ${settings.maxTokens}`));
  console.log(chalk.gray(`  Processing Batch Size: ${settings.processingBatchSize}`));
  console.log('');

  // JSON output for programmatic use
  const jsonOutput = {
    timestamp: new Date().toISOString(),
    hardware: {
      provider: hw.provider,
      device: hw.deviceName,
      cudaVersion: hw.cudaVersion,
      memoryTotal: hw.memoryTotal,
    },
    config: {
      maxTokens: settings.maxTokens,
      processingBatchSize: settings.processingBatchSize,
      iterations,
    },
    results,
  };

  console.log(chalk.gray('JSON Output:'));
  console.log(JSON.stringify(jsonOutput, null, 2));
  console.log('');
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  try {
    // Set provider if specified
    if (options.provider) {
      process.env.EXECUTION_PROVIDER = options.provider;
      if (options.provider === 'cpu') {
        process.env.FORCE_CPU = 'true';
      }
    }

    if (options.check) {
      await checkModels();
    } else if (options.benchmark) {
      await runBenchmarks();
    } else {
      program.help();
    }
  } catch (error) {
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
