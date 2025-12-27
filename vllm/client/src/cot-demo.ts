/**
 * Chain-of-Thought Reasoning Verification Demo
 *
 * Demonstrates GPT-OSS 20B's reasoning capabilities by showing:
 * - The model's internal reasoning process
 * - The final response
 * - Verification of logical steps
 *
 * Runs fully automatically - no user input required.
 *
 * Usage:
 *   bun run cot-demo.ts                     # Run full interactive demo
 *   bun run cot-demo.ts --prompt "..."      # Run with custom prompt
 *   bun run cot-demo.ts --test              # Run all verification tests
 *   bun run cot-demo.ts --test math         # Run specific test
 *   bun run cot-demo.ts --quick             # Run quick 2-example demo
 */

import axios from 'axios';
import chalk from 'chalk';

const INFERENCE_URL = process.env.INFERENCE_URL || 'http://localhost:8004';

// ============================================================
// Types
// ============================================================

interface GptOssChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      reasoning?: string;
      reasoning_content?: string;
      tool_calls?: any[];
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ReasoningResult {
  prompt: string;
  reasoning: string;
  response: string | null;
  finishReason: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  timing: {
    startTime: number;
    endTime: number;
    durationMs: number;
  };
}

interface DemoExample {
  category: string;
  prompt: string;
  maxTokens: number;
  expectedAnswer?: string;
  expectedContains?: string[];
  description: string;
}

// ============================================================
// Demo Examples - Fully Automatic
// ============================================================

const DEMO_EXAMPLES: DemoExample[] = [
  {
    category: 'Mathematical Reasoning',
    description: 'Classic cognitive reflection test - the bat and ball problem',
    prompt: `A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

Solve this step by step, showing your algebra.`,
    maxTokens: 800,
    expectedAnswer: '$0.05',
  },
  {
    category: 'Logical Deduction',
    description: 'Mislabeled boxes puzzle requiring systematic reasoning',
    prompt: `Three boxes are labeled "Apples", "Oranges", and "Mixed". ALL labels are wrong.

You can pick ONE fruit from ONE box to correctly relabel all boxes.

Which box do you pick from, and how do you figure out all labels? Explain your reasoning.`,
    maxTokens: 1000,
    expectedContains: ['mixed', 'pick'],
  },
  {
    category: 'Code Analysis',
    description: 'Python bug identification and correction',
    prompt: `This Python function has a bug. Find it and fix it:

def is_palindrome(s):
    return s == s.reverse()

What's wrong and what's the correct code?`,
    maxTokens: 600,
    expectedContains: ['[::-1]', 'reversed', 'list'],
  },
  {
    category: 'Legal Reasoning',
    description: 'Contract law elements analysis',
    prompt: `What are the four essential elements required for a valid contract? List each element with a one-sentence explanation.`,
    maxTokens: 800,
    expectedContains: ['offer', 'acceptance', 'consideration'],
  },
];

const QUICK_EXAMPLES: DemoExample[] = [
  {
    category: 'Math Problem',
    description: 'Average speed calculation',
    prompt: `A car drives 60 miles at 30 mph, then 60 miles at 60 mph. What is the average speed for the entire 120-mile trip?

Hint: Average speed is total distance divided by total time.`,
    maxTokens: 600,
    expectedAnswer: '40',
  },
  {
    category: 'Logic Puzzle',
    description: 'Classic "all but" language interpretation',
    prompt: `A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?`,
    maxTokens: 400,
    expectedAnswer: '9',
  },
];

const TEST_CASES = {
  math: {
    name: 'Mathematical Reasoning',
    prompt: `A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?

Think through this step by step and give me the final answer.`,
    maxTokens: 600,
    expectedAnswer: '9',
  },
  logic: {
    name: 'Logical Deduction',
    prompt: `If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?

Explain your reasoning carefully.`,
    maxTokens: 800,
    expectedContains: ['cannot conclude', 'no', 'not necessarily', 'invalid'],
  },
  code: {
    name: 'Code Analysis',
    prompt: `What does this code do and what will it print?

\`\`\`python
def mystery(n):
    if n <= 1:
        return n
    return mystery(n-1) + mystery(n-2)

print(mystery(6))
\`\`\`

Trace through the execution.`,
    maxTokens: 1000,
    expectedAnswer: '8',
  },
  legal: {
    name: 'Legal Analysis',
    prompt: `In contract law, what are the essential elements required for a valid contract?

List and briefly explain each element.`,
    maxTokens: 800,
    expectedContains: ['offer', 'acceptance', 'consideration'],
  },
  verification: {
    name: 'Self-Verification',
    prompt: `I claim that 15 + 28 = 42.

Am I correct? Show your work to verify.`,
    maxTokens: 500,
    expectedAnswer: '43',
  },
};

// ============================================================
// GPT-OSS Client
// ============================================================

async function getModelId(): Promise<string> {
  try {
    const response = await axios.get(`${INFERENCE_URL}/v1/models`, { timeout: 5000 });
    if (response.data.data && response.data.data.length > 0) {
      return response.data.data[0].id;
    }
  } catch (error) {
    throw new Error(`Cannot connect to inference service at ${INFERENCE_URL}`);
  }
  throw new Error('No model available');
}

async function runReasoningQuery(
  prompt: string,
  systemPrompt?: string,
  maxTokens: number = 1024
): Promise<ReasoningResult> {
  const modelId = await getModelId();

  const messages: Array<{ role: string; content: string }> = [];
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  const startTime = performance.now();

  const response = await axios.post<GptOssChatResponse>(
    `${INFERENCE_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages,
      max_tokens: maxTokens,
      temperature: 0.3,
    },
    { timeout: 120000 }
  );

  const endTime = performance.now();
  const choice = response.data.choices[0];

  return {
    prompt,
    reasoning: choice.message.reasoning_content || choice.message.reasoning || '',
    response: choice.message.content,
    finishReason: choice.finish_reason,
    usage: {
      promptTokens: response.data.usage.prompt_tokens,
      completionTokens: response.data.usage.completion_tokens,
      totalTokens: response.data.usage.total_tokens,
    },
    timing: {
      startTime,
      endTime,
      durationMs: endTime - startTime,
    },
  };
}

// ============================================================
// Display Functions
// ============================================================

function printHeader() {
  console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
  console.log(chalk.cyan.bold('       GPT-OSS 20B Chain-of-Thought Reasoning Demo'));
  console.log(chalk.cyan.bold('='.repeat(80) + '\n'));
}

function printSection(title: string) {
  console.log(chalk.white.bold('\n' + '-'.repeat(60)));
  console.log(chalk.white.bold(`  ${title}`));
  console.log(chalk.white.bold('-'.repeat(60) + '\n'));
}

function formatReasoning(reasoning: string): string {
  if (!reasoning) return chalk.dim('  (no reasoning provided)');
  return reasoning
    .split('\n')
    .map(line => chalk.yellow('  | ') + chalk.dim(line))
    .join('\n');
}

function formatResponse(response: string | null): string {
  if (!response) return chalk.dim('  (no final response - see reasoning above)');
  return response
    .split('\n')
    .map(line => chalk.green('  > ') + chalk.white(line))
    .join('\n');
}

function printResult(result: ReasoningResult, showFullMetrics: boolean = true) {
  // Reasoning (Chain of Thought)
  if (result.reasoning) {
    printSection('Model Reasoning (Chain-of-Thought)');
    console.log(formatReasoning(result.reasoning));
  }

  // Final Response
  printSection('Final Response');
  console.log(formatResponse(result.response));

  // Metrics
  if (showFullMetrics) {
    printSection('Metrics');
    console.log(chalk.gray(`  Duration:       ${result.timing.durationMs.toFixed(0)}ms`));
    console.log(chalk.gray(`  Prompt tokens:  ${result.usage.promptTokens}`));
    console.log(chalk.gray(`  Output tokens:  ${result.usage.completionTokens}`));
    console.log(chalk.gray(`  Total tokens:   ${result.usage.totalTokens}`));
    console.log(chalk.gray(`  Speed:          ${(result.usage.completionTokens / (result.timing.durationMs / 1000)).toFixed(1)} tokens/sec`));
    console.log(chalk.gray(`  Finish reason:  ${result.finishReason}`));
  } else {
    console.log(chalk.gray(`\n  [${result.timing.durationMs.toFixed(0)}ms | ${result.usage.completionTokens} tokens | ${(result.usage.completionTokens / (result.timing.durationMs / 1000)).toFixed(1)} tok/s]`));
  }
}

function verifyResult(
  result: ReasoningResult,
  expectedAnswer?: string,
  expectedContains?: string[]
): { passed: boolean; details: string } {
  const fullText = `${result.reasoning} ${result.response || ''}`.toLowerCase();

  if (expectedAnswer) {
    const normalizedExpected = expectedAnswer.toLowerCase().replace(/[$,]/g, '');
    const found = fullText.includes(normalizedExpected) ||
                  fullText.includes(expectedAnswer.toLowerCase());
    return {
      passed: found,
      details: found
        ? `Found expected answer "${expectedAnswer}"`
        : `Expected "${expectedAnswer}" not found in response`,
    };
  }

  if (expectedContains) {
    const found = expectedContains.filter(term => fullText.includes(term.toLowerCase()));
    const missing = expectedContains.filter(term => !fullText.includes(term.toLowerCase()));

    if (missing.length === 0) {
      return { passed: true, details: `Found all expected terms: ${found.join(', ')}` };
    }
    return {
      passed: found.length > 0,
      details: `Found: ${found.join(', ')}${missing.length > 0 ? ` | Missing: ${missing.join(', ')}` : ''}`,
    };
  }

  return { passed: true, details: 'No verification criteria' };
}

// ============================================================
// Main Demo Functions
// ============================================================

async function runSingleDemo(prompt: string) {
  printHeader();

  console.log(chalk.white('Connecting to GPT-OSS 20B...'));

  try {
    const modelId = await getModelId();
    console.log(chalk.green(`Connected to model: ${modelId}\n`));

    console.log(chalk.white('Running chain-of-thought query...'));

    printSection('User Prompt');
    console.log(chalk.white(prompt));

    const result = await runReasoningQuery(prompt, undefined, 1500);
    printResult(result);

    console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
    console.log(chalk.green.bold('                    Demo Complete'));
    console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

  } catch (error) {
    console.log(chalk.red(`\nError: ${error instanceof Error ? error.message : error}`));
    process.exit(1);
  }
}

async function runTestSuite(testName?: string) {
  printHeader();

  console.log(chalk.white('Connecting to GPT-OSS 20B...'));

  try {
    const modelId = await getModelId();
    console.log(chalk.green(`Connected to model: ${modelId}\n`));

    const tests = testName
      ? { [testName]: TEST_CASES[testName as keyof typeof TEST_CASES] }
      : TEST_CASES;

    if (testName && !TEST_CASES[testName as keyof typeof TEST_CASES]) {
      console.log(chalk.red(`Unknown test: ${testName}`));
      console.log(chalk.gray(`Available tests: ${Object.keys(TEST_CASES).join(', ')}`));
      process.exit(1);
    }

    const results: Array<{ name: string; passed: boolean; duration: number; details: string }> = [];

    for (const [key, test] of Object.entries(tests)) {
      console.log(chalk.cyan.bold(`\n${'#'.repeat(80)}`));
      console.log(chalk.cyan.bold(`# Test: ${test.name}`));
      console.log(chalk.cyan.bold(`${'#'.repeat(80)}`));

      printSection('User Prompt');
      console.log(chalk.white(test.prompt));

      console.log(chalk.gray('\nRunning query...'));
      const result = await runReasoningQuery(test.prompt, undefined, test.maxTokens);

      printResult(result);

      // Verify
      const verification = verifyResult(
        result,
        (test as any).expectedAnswer,
        (test as any).expectedContains
      );

      printSection('Verification');
      if (verification.passed) {
        console.log(chalk.green.bold('  PASSED'));
      } else {
        console.log(chalk.red.bold('  FAILED'));
      }
      console.log(chalk.gray(`  ${verification.details}`));

      results.push({
        name: test.name,
        passed: verification.passed,
        duration: result.timing.durationMs,
        details: verification.details,
      });
    }

    // Summary
    console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
    console.log(chalk.cyan.bold('                       Test Summary'));
    console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;
    const totalTime = results.reduce((sum, r) => sum + r.duration, 0);

    for (const r of results) {
      const status = r.passed ? chalk.green.bold('PASS') : chalk.red.bold('FAIL');
      console.log(`  ${status}  ${r.name.padEnd(30)} ${chalk.gray(`${r.duration.toFixed(0)}ms`)}`);
    }

    console.log('');
    console.log(chalk.white.bold(`  Results: ${passed}/${results.length} passed`));
    console.log(chalk.gray(`  Total time: ${(totalTime / 1000).toFixed(1)}s`));

    console.log(chalk.cyan.bold('\n' + '='.repeat(80) + '\n'));

    process.exit(failed > 0 ? 1 : 0);

  } catch (error) {
    console.log(chalk.red(`\nError: ${error instanceof Error ? error.message : error}`));
    process.exit(1);
  }
}

async function runFullDemo(quick: boolean = false) {
  printHeader();

  const examples = quick ? QUICK_EXAMPLES : DEMO_EXAMPLES;

  console.log(chalk.white.bold('GPT-OSS 20B Chain-of-Thought Demonstration\n'));
  console.log(chalk.gray('This demo runs automatically through multiple reasoning examples.'));
  console.log(chalk.gray('Each example shows the model\'s internal reasoning process'));
  console.log(chalk.gray('followed by its final response.\n'));

  try {
    const modelId = await getModelId();
    console.log(chalk.green(`Connected to: ${modelId}`));
    console.log(chalk.gray(`Examples to run: ${examples.length}\n`));

    const results: Array<{
      category: string;
      passed: boolean;
      duration: number;
      tokensPerSec: number;
    }> = [];

    for (let i = 0; i < examples.length; i++) {
      const example = examples[i];

      console.log(chalk.cyan.bold(`\n${'='.repeat(80)}`));
      console.log(chalk.cyan.bold(`  Example ${i + 1}/${examples.length}: ${example.category}`));
      console.log(chalk.cyan.bold(`${'='.repeat(80)}`));
      console.log(chalk.dim(`  ${example.description}\n`));

      printSection('User Prompt');
      console.log(chalk.white(example.prompt));

      console.log(chalk.gray('\n  Processing...\n'));

      const result = await runReasoningQuery(example.prompt, undefined, example.maxTokens);

      printResult(result, true);

      // Verify if criteria provided
      if (example.expectedAnswer || example.expectedContains) {
        const verification = verifyResult(result, example.expectedAnswer, example.expectedContains);
        printSection('Verification');
        if (verification.passed) {
          console.log(chalk.green.bold('  PASSED'));
        } else {
          console.log(chalk.yellow.bold('  PARTIAL'));
        }
        console.log(chalk.gray(`  ${verification.details}`));

        results.push({
          category: example.category,
          passed: verification.passed,
          duration: result.timing.durationMs,
          tokensPerSec: result.usage.completionTokens / (result.timing.durationMs / 1000),
        });
      } else {
        results.push({
          category: example.category,
          passed: true,
          duration: result.timing.durationMs,
          tokensPerSec: result.usage.completionTokens / (result.timing.durationMs / 1000),
        });
      }
    }

    // Final Summary
    console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
    console.log(chalk.cyan.bold('                       Demo Summary'));
    console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

    console.log(chalk.white.bold('  Results by Example:\n'));
    for (const r of results) {
      const status = r.passed ? chalk.green('OK') : chalk.yellow('--');
      console.log(`    ${status}  ${r.category.padEnd(25)} ${chalk.gray(`${(r.duration / 1000).toFixed(1)}s`)} ${chalk.dim(`(${r.tokensPerSec.toFixed(0)} tok/s)`)}`);
    }

    const totalTime = results.reduce((sum, r) => sum + r.duration, 0);
    const avgTokPerSec = results.reduce((sum, r) => sum + r.tokensPerSec, 0) / results.length;
    const passedCount = results.filter(r => r.passed).length;

    console.log('');
    console.log(chalk.white.bold('  Overall:'));
    console.log(chalk.gray(`    Examples completed: ${results.length}`));
    console.log(chalk.gray(`    Verification passed: ${passedCount}/${results.length}`));
    console.log(chalk.gray(`    Total time: ${(totalTime / 1000).toFixed(1)}s`));
    console.log(chalk.gray(`    Avg throughput: ${avgTokPerSec.toFixed(1)} tokens/sec`));

    console.log(chalk.cyan.bold('\n' + '='.repeat(80)));
    console.log(chalk.green.bold('              Chain-of-Thought Demo Complete'));
    console.log(chalk.cyan.bold('='.repeat(80) + '\n'));

  } catch (error) {
    console.log(chalk.red(`\nError: ${error instanceof Error ? error.message : error}`));
    process.exit(1);
  }
}

// ============================================================
// CLI Entry Point
// ============================================================

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
${chalk.cyan.bold('GPT-OSS Chain-of-Thought Reasoning Demo')}

${chalk.white.bold('Usage:')}
  bun run cot-demo.ts                     Run full demo (4 examples)
  bun run cot-demo.ts --quick             Run quick demo (2 examples)
  bun run cot-demo.ts --prompt "..."      Run with custom prompt
  bun run cot-demo.ts --test              Run all verification tests
  bun run cot-demo.ts --test math         Run specific test

${chalk.white.bold('Available tests:')}
  ${Object.entries(TEST_CASES).map(([key, val]) => `${key.padEnd(15)} ${val.name}`).join('\n  ')}

${chalk.white.bold('Options:')}
  --prompt <text>   Custom prompt to run
  --test [name]     Run verification tests
  --quick           Run abbreviated 2-example demo
  --help, -h        Show this help

${chalk.white.bold('Environment:')}
  INFERENCE_URL     GPT-OSS endpoint (default: http://localhost:8004)

${chalk.white.bold('Features:')}
  - Shows model's internal reasoning process (chain-of-thought)
  - Displays final response separately
  - Verifies answers against expected results
  - Reports timing and throughput metrics
`);
    process.exit(0);
  }

  // Custom prompt
  const promptIdx = args.indexOf('--prompt');
  if (promptIdx !== -1 && args[promptIdx + 1]) {
    await runSingleDemo(args[promptIdx + 1]);
    return;
  }

  // Test suite
  const testIdx = args.indexOf('--test');
  if (testIdx !== -1) {
    const testName = args[testIdx + 1];
    await runTestSuite(testName && !testName.startsWith('--') ? testName : undefined);
    return;
  }

  // Quick demo
  if (args.includes('--quick')) {
    await runFullDemo(true);
    return;
  }

  // Default: full demo
  await runFullDemo(false);
}

main().catch(console.error);
