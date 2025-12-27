/**
 * Embedding Load Balancer Stress Test
 *
 * Stress tests the embedding service with high concurrency to measure:
 *   - Maximum throughput under load
 *   - Latency distribution (p50, p95, p99)
 *   - Load balancer effectiveness (single vs LB comparison)
 *   - Error rates and stability
 *
 * Usage:
 *   bun run stress                          # Default: 50 concurrent, 30s
 *   bun run stress --concurrency 100        # 100 concurrent requests
 *   bun run stress --duration 60            # 60 second test
 *   bun run stress --ramp-up 10             # 10 second ramp-up
 *   bun run stress --compare                # Compare single vs LB
 */

import axios from 'axios';
import chalk from 'chalk';

// ============================================================================
// Configuration
// ============================================================================

const SINGLE_ENDPOINT = process.env.EMBEDDINGS_URL || 'http://localhost:8001';
const LB_ENDPOINT = process.env.LB_URL || 'http://localhost:8000';

interface StressConfig {
  concurrency: number;
  duration: number;
  rampUp: number;
  compare: boolean;
  endpoint: string;
}

interface RequestResult {
  success: boolean;
  latencyMs: number;
  timestamp: number;
  error?: string;
}

interface StressResults {
  endpoint: string;
  config: StressConfig;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  totalDuration: number;
  throughput: number;
  latency: {
    min: number;
    max: number;
    avg: number;
    p50: number;
    p95: number;
    p99: number;
  };
  errorRate: number;
  requestsPerSecond: number[];
}

// ============================================================================
// Utilities
// ============================================================================

function parseArgs(): StressConfig {
  const args = process.argv.slice(2);
  const config: StressConfig = {
    concurrency: 50,
    duration: 30,
    rampUp: 5,
    compare: false,
    endpoint: SINGLE_ENDPOINT,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--concurrency':
      case '-c':
        config.concurrency = parseInt(args[++i]) || 50;
        break;
      case '--duration':
      case '-d':
        config.duration = parseInt(args[++i]) || 30;
        break;
      case '--ramp-up':
      case '-r':
        config.rampUp = parseInt(args[++i]) || 5;
        break;
      case '--compare':
        config.compare = true;
        break;
      case '--lb':
        config.endpoint = LB_ENDPOINT;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
    }
  }

  return config;
}

function printHelp() {
  console.log(`
${chalk.cyan.bold('Embedding Load Balancer Stress Test')}

${chalk.white.bold('Usage:')}
  bun run stress [options]

${chalk.white.bold('Options:')}
  -c, --concurrency <n>   Concurrent requests (default: 50)
  -d, --duration <s>      Test duration in seconds (default: 30)
  -r, --ramp-up <s>       Ramp-up time in seconds (default: 5)
  --compare               Compare single endpoint vs load balancer
  --lb                    Use load balancer endpoint
  -h, --help              Show this help

${chalk.white.bold('Examples:')}
  bun run stress                          # Default stress test
  bun run stress -c 100 -d 60             # 100 concurrent for 60s
  bun run stress --compare                # Compare single vs LB
  bun run stress --lb -c 200              # Stress test LB with 200 concurrent
`);
}

async function getModelId(endpoint: string): Promise<string | null> {
  try {
    const response = await axios.get(`${endpoint}/v1/models`, { timeout: 5000 });
    return response.data.data?.[0]?.id || null;
  } catch {
    return null;
  }
}

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function formatNumber(n: number, decimals: number = 1): string {
  return n.toFixed(decimals);
}

// ============================================================================
// Stress Test Engine
// ============================================================================

async function runStressTest(
  endpoint: string,
  modelId: string,
  config: StressConfig,
  onProgress?: (stats: { rps: number; latency: number; errors: number }) => void
): Promise<StressResults> {
  const results: RequestResult[] = [];
  const rpsHistory: number[] = [];
  let activeRequests = 0;
  let currentConcurrency = 0;
  let running = true;

  const testTexts = [
    'Legal document for embedding stress test.',
    'Contract analysis and semantic search verification.',
    'Court filing with multiple paragraphs of legal terminology.',
    'Brief summary of the plaintiff motion for summary judgment.',
    'Intellectual property rights and patent infringement claims.',
  ];

  const startTime = Date.now();
  const endTime = startTime + config.duration * 1000;
  const rampUpEnd = startTime + config.rampUp * 1000;

  // Worker function
  async function worker() {
    while (running && Date.now() < endTime) {
      if (activeRequests >= currentConcurrency) {
        await new Promise(r => setTimeout(r, 1));
        continue;
      }

      activeRequests++;
      const reqStart = Date.now();
      const text = testTexts[Math.floor(Math.random() * testTexts.length)];

      try {
        await axios.post(
          `${endpoint}/v1/embeddings`,
          { model: modelId, input: text },
          { timeout: 30000 }
        );
        results.push({
          success: true,
          latencyMs: Date.now() - reqStart,
          timestamp: reqStart,
        });
      } catch (error) {
        results.push({
          success: false,
          latencyMs: Date.now() - reqStart,
          timestamp: reqStart,
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      }

      activeRequests--;
    }
  }

  // Progress reporter
  const progressInterval = setInterval(() => {
    const now = Date.now();
    const elapsed = (now - startTime) / 1000;

    // Calculate RPS for last second
    const lastSecond = results.filter(r => r.timestamp > now - 1000);
    const rps = lastSecond.length;
    rpsHistory.push(rps);

    // Calculate current latency
    const recentLatencies = lastSecond.map(r => r.latencyMs);
    const avgLatency = recentLatencies.length > 0
      ? recentLatencies.reduce((a, b) => a + b, 0) / recentLatencies.length
      : 0;

    // Error count
    const errors = results.filter(r => !r.success).length;

    // Update concurrency during ramp-up
    if (now < rampUpEnd) {
      const rampProgress = (now - startTime) / (config.rampUp * 1000);
      currentConcurrency = Math.ceil(config.concurrency * rampProgress);
    } else {
      currentConcurrency = config.concurrency;
    }

    if (onProgress) {
      onProgress({ rps, latency: avgLatency, errors });
    }
  }, 1000);

  // Start workers
  const workers: Promise<void>[] = [];
  for (let i = 0; i < config.concurrency; i++) {
    workers.push(worker());
  }

  // Wait for completion
  await Promise.all(workers);
  running = false;
  clearInterval(progressInterval);

  // Calculate results
  const successfulResults = results.filter(r => r.success);
  const latencies = successfulResults.map(r => r.latencyMs);
  const totalDuration = (Date.now() - startTime) / 1000;

  return {
    endpoint,
    config,
    totalRequests: results.length,
    successfulRequests: successfulResults.length,
    failedRequests: results.filter(r => !r.success).length,
    totalDuration,
    throughput: results.length / totalDuration,
    latency: {
      min: latencies.length > 0 ? Math.min(...latencies) : 0,
      max: latencies.length > 0 ? Math.max(...latencies) : 0,
      avg: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
      p50: percentile(latencies, 50),
      p95: percentile(latencies, 95),
      p99: percentile(latencies, 99),
    },
    errorRate: results.length > 0 ? (results.filter(r => !r.success).length / results.length) * 100 : 0,
    requestsPerSecond: rpsHistory,
  };
}

// ============================================================================
// Output Formatting (Benchmark Style)
// ============================================================================

function printHeader() {
  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('            Embedding Load Balancer Stress Test'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));
}

function printConfig(config: StressConfig, modelId: string) {
  console.log(chalk.white.bold('Configuration'));
  console.log(chalk.dim('-'.repeat(50)));
  console.log(chalk.gray(`  Endpoint:     ${config.endpoint}`));
  console.log(chalk.gray(`  Model:        ${modelId}`));
  console.log(chalk.gray(`  Concurrency:  ${config.concurrency}`));
  console.log(chalk.gray(`  Duration:     ${config.duration}s`));
  console.log(chalk.gray(`  Ramp-up:      ${config.rampUp}s`));
  console.log('');
}

function printProgress(elapsed: number, duration: number, stats: { rps: number; latency: number; errors: number }) {
  const pct = Math.min(100, (elapsed / duration) * 100);
  const bar = '='.repeat(Math.floor(pct / 2)) + '>' + ' '.repeat(50 - Math.floor(pct / 2));

  process.stdout.write(`\r  [${bar}] ${pct.toFixed(0)}% | ` +
    chalk.green(`${stats.rps} req/s`) + ' | ' +
    chalk.yellow(`${stats.latency.toFixed(0)}ms`) + ' | ' +
    (stats.errors > 0 ? chalk.red(`${stats.errors} errors`) : chalk.gray('0 errors')) +
    '   ');
}

function printResults(results: StressResults) {
  console.log('\n');
  console.log(chalk.white.bold('Results'));
  console.log(chalk.dim('-'.repeat(50)));

  // Throughput
  console.log(chalk.white(`  Total Requests:     ${results.totalRequests.toLocaleString()}`));
  console.log(chalk.white(`  Successful:         ${results.successfulRequests.toLocaleString()}`));
  if (results.failedRequests > 0) {
    console.log(chalk.red(`  Failed:             ${results.failedRequests.toLocaleString()}`));
  }
  console.log(chalk.white(`  Duration:           ${formatNumber(results.totalDuration)}s`));
  console.log('');

  console.log(chalk.white.bold('Throughput'));
  console.log(chalk.dim('-'.repeat(50)));
  console.log(chalk.green.bold(`  Requests/sec:       ${formatNumber(results.throughput)} req/s`));
  console.log(chalk.gray(`  Peak RPS:           ${Math.max(...results.requestsPerSecond)} req/s`));
  console.log(chalk.gray(`  Avg RPS:            ${formatNumber(results.requestsPerSecond.reduce((a, b) => a + b, 0) / results.requestsPerSecond.length)} req/s`));
  console.log('');

  console.log(chalk.white.bold('Latency'));
  console.log(chalk.dim('-'.repeat(50)));
  console.log(chalk.white(`  Min:                ${formatNumber(results.latency.min)}ms`));
  console.log(chalk.white(`  Avg:                ${formatNumber(results.latency.avg)}ms`));
  console.log(chalk.white(`  Max:                ${formatNumber(results.latency.max)}ms`));
  console.log(chalk.yellow(`  p50:                ${formatNumber(results.latency.p50)}ms`));
  console.log(chalk.yellow(`  p95:                ${formatNumber(results.latency.p95)}ms`));
  console.log(chalk.red(`  p99:                ${formatNumber(results.latency.p99)}ms`));
  console.log('');

  if (results.errorRate > 0) {
    console.log(chalk.white.bold('Errors'));
    console.log(chalk.dim('-'.repeat(50)));
    console.log(chalk.red(`  Error Rate:         ${formatNumber(results.errorRate)}%`));
    console.log('');
  }
}

function printComparison(single: StressResults, lb: StressResults) {
  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('                    Comparison Summary'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

  const throughputImprovement = ((lb.throughput - single.throughput) / single.throughput) * 100;
  const latencyImprovement = ((single.latency.avg - lb.latency.avg) / single.latency.avg) * 100;

  console.log(chalk.white.bold('                          Single         Load Balanced   Improvement'));
  console.log(chalk.dim('-'.repeat(70)));
  console.log(`  Throughput (req/s)      ${formatNumber(single.throughput).padStart(8)}         ${formatNumber(lb.throughput).padStart(8)}        ${throughputImprovement > 0 ? chalk.green('+' + formatNumber(throughputImprovement) + '%') : chalk.red(formatNumber(throughputImprovement) + '%')}`);
  console.log(`  Avg Latency (ms)        ${formatNumber(single.latency.avg).padStart(8)}         ${formatNumber(lb.latency.avg).padStart(8)}        ${latencyImprovement > 0 ? chalk.green('+' + formatNumber(latencyImprovement) + '%') : chalk.red(formatNumber(latencyImprovement) + '%')}`);
  console.log(`  p95 Latency (ms)        ${formatNumber(single.latency.p95).padStart(8)}         ${formatNumber(lb.latency.p95).padStart(8)}`);
  console.log(`  p99 Latency (ms)        ${formatNumber(single.latency.p99).padStart(8)}         ${formatNumber(lb.latency.p99).padStart(8)}`);
  console.log(`  Error Rate              ${formatNumber(single.errorRate).padStart(7)}%         ${formatNumber(lb.errorRate).padStart(7)}%`);
  console.log('');
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const config = parseArgs();

  printHeader();

  // Check services
  console.log(chalk.white.bold('Service Discovery'));
  console.log(chalk.dim('-'.repeat(50)));

  const singleModel = await getModelId(SINGLE_ENDPOINT);
  const lbModel = await getModelId(LB_ENDPOINT);

  console.log(singleModel
    ? chalk.green(`  [OK] Single (${SINGLE_ENDPOINT})`) + chalk.dim(` - ${singleModel}`)
    : chalk.red(`  [X] Single endpoint not available`));
  console.log(lbModel
    ? chalk.green(`  [OK] Load Balanced (${LB_ENDPOINT})`) + chalk.dim(` - Traefik LB`)
    : chalk.dim(`  [--] Load Balanced: not available`));
  console.log('');

  if (!singleModel && !lbModel) {
    console.log(chalk.red('No embedding services available. Start with: make up-scaled'));
    process.exit(1);
  }

  if (config.compare) {
    if (!singleModel || !lbModel) {
      console.log(chalk.red('Both single and LB endpoints required for comparison.'));
      console.log(chalk.gray('Start with: make up-scaled'));
      process.exit(1);
    }

    // Run single endpoint test
    console.log(chalk.cyan.bold('\n--- Testing Single Endpoint ---\n'));
    printConfig({ ...config, endpoint: SINGLE_ENDPOINT }, singleModel);

    console.log(chalk.white.bold('Progress'));
    console.log(chalk.dim('-'.repeat(50)));
    let elapsed = 0;
    const singleResults = await runStressTest(SINGLE_ENDPOINT, singleModel, config, (stats) => {
      elapsed++;
      printProgress(elapsed, config.duration, stats);
    });
    printResults(singleResults);

    // Run load balanced test
    console.log(chalk.cyan.bold('\n--- Testing Load Balanced Endpoint ---\n'));
    printConfig({ ...config, endpoint: LB_ENDPOINT }, lbModel!);

    console.log(chalk.white.bold('Progress'));
    console.log(chalk.dim('-'.repeat(50)));
    elapsed = 0;
    const lbResults = await runStressTest(LB_ENDPOINT, lbModel!, config, (stats) => {
      elapsed++;
      printProgress(elapsed, config.duration, stats);
    });
    printResults(lbResults);

    // Print comparison
    printComparison(singleResults, lbResults);

  } else {
    // Single test
    const endpoint = config.endpoint;
    const modelId = endpoint === LB_ENDPOINT ? lbModel : singleModel;

    if (!modelId) {
      console.log(chalk.red(`Endpoint ${endpoint} not available.`));
      process.exit(1);
    }

    printConfig(config, modelId);

    console.log(chalk.white.bold('Progress'));
    console.log(chalk.dim('-'.repeat(50)));

    let elapsed = 0;
    const results = await runStressTest(endpoint, modelId, config, (stats) => {
      elapsed++;
      printProgress(elapsed, config.duration, stats);
    });

    printResults(results);
  }

  console.log(chalk.cyan.bold('='.repeat(70)));
  console.log(chalk.green.bold('                    Stress Test Complete'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));
}

main().catch(error => {
  console.error(chalk.red(`\nFatal error: ${error.message}`));
  process.exit(1);
});
