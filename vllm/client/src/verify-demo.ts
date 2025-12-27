/**
 * OCR & Embedding Verification Demo
 *
 * Verifies that OCR and embedding/search actually work correctly:
 *   1. OCR: Generates test images with known text, verifies extraction
 *   2. Embeddings: Tests semantic similarity with known pairs
 *   3. Search: Verifies ranking of similar vs dissimilar documents
 *
 * Usage:
 *   bun run verify              # Run all verification tests
 *   bun run verify --ocr        # OCR verification only
 *   bun run verify --embedding  # Embedding verification only
 *   bun run verify --search     # Search verification only
 */

import axios from 'axios';
import chalk from 'chalk';
import sharp from 'sharp';
import { join } from 'path';
import { execSync } from 'child_process';

// Service URLs
const EMBEDDINGS_URL = process.env.EMBEDDINGS_URL || 'http://localhost:8001';
const OCR_URL = process.env.OCR_URL || 'http://localhost:8003';

// ============================================================
// Types
// ============================================================

interface VerificationResult {
  test: string;
  category: string;
  passed: boolean;
  score?: number;
  expected?: string;
  actual?: string;
  details: string;
  timeMs: number;
}

interface EmbeddingResponse {
  data: Array<{ embedding: number[] }>;
}

interface ChatResponse {
  choices: Array<{ message: { content: string } }>;
}

// ============================================================
// Utilities
// ============================================================

async function getModelId(url: string): Promise<string | null> {
  try {
    const response = await axios.get(`${url}/v1/models`, { timeout: 5000 });
    return response.data.data?.[0]?.id || null;
  } catch {
    return null;
  }
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

function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      matrix[i][j] = b[i - 1] === a[j - 1]
        ? matrix[i - 1][j - 1]
        : Math.min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1);
    }
  }
  return matrix[b.length][a.length];
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function textSimilarity(expected: string, actual: string): number {
  const normExpected = normalizeText(expected);
  const normActual = normalizeText(actual);
  if (normExpected === normActual) return 1.0;
  const maxLen = Math.max(normExpected.length, normActual.length);
  if (maxLen === 0) return 1.0;
  const distance = levenshteinDistance(normExpected, normActual);
  return Math.max(0, 1 - distance / maxLen);
}

// ============================================================
// Test Image Generation
// ============================================================

async function createTextImage(
  text: string,
  width: number = 800,
  height: number = 200,
  fontSize: number = 48
): Promise<Buffer> {
  // Create SVG with text
  const svg = `
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
            font-family="Arial, sans-serif" font-size="${fontSize}" fill="black">
        ${text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}
      </text>
    </svg>
  `;

  return sharp(Buffer.from(svg))
    .png()
    .toBuffer();
}

async function createMultiLineImage(
  lines: string[],
  width: number = 800,
  height: number = 400,
  fontSize: number = 32
): Promise<Buffer> {
  const lineHeight = fontSize * 1.5;
  const startY = (height - lines.length * lineHeight) / 2 + fontSize;

  const textElements = lines.map((line, i) =>
    `<text x="50" y="${startY + i * lineHeight}" font-family="Arial, sans-serif" font-size="${fontSize}" fill="black">${line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</text>`
  ).join('\n');

  const svg = `
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      ${textElements}
    </svg>
  `;

  return sharp(Buffer.from(svg))
    .png()
    .toBuffer();
}

// ============================================================
// OCR Verification
// ============================================================

async function runOcr(imageBase64: string, modelId: string): Promise<string> {
  const response = await axios.post<ChatResponse>(
    `${OCR_URL}/v1/chat/completions`,
    {
      model: modelId,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: 'Extract all text from this image. Return only the text, nothing else.' },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${imageBase64}` } },
        ],
      }],
      max_tokens: 1024,
      temperature: 0.1,
    },
    { timeout: 60000 }
  );
  return response.data.choices[0].message.content || '';
}

async function verifyOcr(): Promise<VerificationResult[]> {
  const results: VerificationResult[] = [];

  const modelId = await getModelId(OCR_URL);
  if (!modelId) {
    return [{
      test: 'OCR Service',
      category: 'OCR',
      passed: false,
      details: 'OCR service not available',
      timeMs: 0,
    }];
  }

  console.log(chalk.white.bold('\n  OCR Verification Tests\n'));
  console.log(chalk.gray(`  Model: ${modelId}\n`));

  // Test cases with expected text
  const testCases = [
    {
      name: 'Simple Text',
      text: 'Hello World',
      threshold: 0.8,
    },
    {
      name: 'Numbers',
      text: '12345 67890',
      threshold: 0.8,
    },
    {
      name: 'Legal Citation',
      text: 'Brown v. Board of Education, 347 U.S. 483 (1954)',
      threshold: 0.6,
    },
    {
      name: 'Mixed Case',
      text: 'The Quick Brown Fox Jumps Over The Lazy Dog',
      threshold: 0.7,
    },
  ];

  for (const tc of testCases) {
    const start = performance.now();
    process.stdout.write(chalk.gray(`    ${tc.name}: `));

    try {
      const imageBuffer = await createTextImage(tc.text, 800, 150, 40);
      const imageBase64 = imageBuffer.toString('base64');
      const ocrResult = await runOcr(imageBase64, modelId);
      const elapsed = performance.now() - start;

      const similarity = textSimilarity(tc.text, ocrResult);
      const passed = similarity >= tc.threshold;

      if (passed) {
        console.log(chalk.green(`PASS`) + chalk.gray(` (${(similarity * 100).toFixed(0)}% match, ${elapsed.toFixed(0)}ms)`));
      } else {
        console.log(chalk.red(`FAIL`) + chalk.gray(` (${(similarity * 100).toFixed(0)}% match, need ${tc.threshold * 100}%)`));
        console.log(chalk.gray(`      Expected: "${tc.text}"`));
        console.log(chalk.gray(`      Got:      "${ocrResult.slice(0, 80)}"`));
      }

      results.push({
        test: tc.name,
        category: 'OCR',
        passed,
        score: similarity,
        expected: tc.text,
        actual: ocrResult.slice(0, 100),
        details: `${(similarity * 100).toFixed(1)}% text match`,
        timeMs: elapsed,
      });

    } catch (error) {
      const elapsed = performance.now() - start;
      console.log(chalk.red(`ERROR: ${error instanceof Error ? error.message : error}`));
      results.push({
        test: tc.name,
        category: 'OCR',
        passed: false,
        details: `Error: ${error instanceof Error ? error.message : error}`,
        timeMs: elapsed,
      });
    }
  }

  // Multi-line test
  const multiLineStart = performance.now();
  process.stdout.write(chalk.gray(`    Multi-line Document: `));

  try {
    const lines = [
      'UNITED STATES DISTRICT COURT',
      'Case No. 2024-CV-12345',
      'Plaintiff v. Defendant',
    ];
    const imageBuffer = await createMultiLineImage(lines, 800, 300, 28);
    const imageBase64 = imageBuffer.toString('base64');
    const ocrResult = await runOcr(imageBase64, modelId);
    const elapsed = performance.now() - multiLineStart;

    // Check if key phrases are found
    const keyPhrases = ['district court', 'case', 'plaintiff', 'defendant'];
    const foundPhrases = keyPhrases.filter(p => ocrResult.toLowerCase().includes(p));
    const phraseScore = foundPhrases.length / keyPhrases.length;
    const passed = phraseScore >= 0.5;

    if (passed) {
      console.log(chalk.green(`PASS`) + chalk.gray(` (${foundPhrases.length}/${keyPhrases.length} phrases, ${elapsed.toFixed(0)}ms)`));
    } else {
      console.log(chalk.red(`FAIL`) + chalk.gray(` (only ${foundPhrases.length}/${keyPhrases.length} phrases found)`));
    }

    results.push({
      test: 'Multi-line Document',
      category: 'OCR',
      passed,
      score: phraseScore,
      details: `Found ${foundPhrases.length}/${keyPhrases.length} key phrases`,
      timeMs: elapsed,
    });

  } catch (error) {
    console.log(chalk.red(`ERROR`));
    results.push({
      test: 'Multi-line Document',
      category: 'OCR',
      passed: false,
      details: `Error: ${error instanceof Error ? error.message : error}`,
      timeMs: performance.now() - multiLineStart,
    });
  }

  return results;
}

// ============================================================
// Embedding Verification
// ============================================================

async function getEmbedding(text: string, modelId: string): Promise<number[]> {
  const response = await axios.post<EmbeddingResponse>(
    `${EMBEDDINGS_URL}/v1/embeddings`,
    { model: modelId, input: text },
    { timeout: 30000 }
  );
  return response.data.data[0].embedding;
}

async function verifyEmbeddings(): Promise<VerificationResult[]> {
  const results: VerificationResult[] = [];

  const modelId = await getModelId(EMBEDDINGS_URL);
  if (!modelId) {
    return [{
      test: 'Embeddings Service',
      category: 'Embeddings',
      passed: false,
      details: 'Embeddings service not available',
      timeMs: 0,
    }];
  }

  console.log(chalk.white.bold('\n  Embedding Verification Tests\n'));
  console.log(chalk.gray(`  Model: ${modelId}\n`));

  // Test 1: Similar sentences should have high similarity
  const similarPairs = [
    {
      name: 'Semantic Similarity',
      a: 'The court ruled in favor of the plaintiff.',
      b: 'The judge decided the case for the plaintiff.',
      minSim: 0.7,
    },
    {
      name: 'Legal Synonyms',
      a: 'The defendant filed a motion to dismiss.',
      b: 'The accused submitted a request for case dismissal.',
      minSim: 0.6,
    },
    {
      name: 'Identical Text',
      a: 'Contract law governs agreements between parties.',
      b: 'Contract law governs agreements between parties.',
      minSim: 0.99,
    },
  ];

  for (const pair of similarPairs) {
    const start = performance.now();
    process.stdout.write(chalk.gray(`    ${pair.name}: `));

    try {
      const [embA, embB] = await Promise.all([
        getEmbedding(pair.a, modelId),
        getEmbedding(pair.b, modelId),
      ]);
      const elapsed = performance.now() - start;

      const similarity = cosineSimilarity(embA, embB);
      const passed = similarity >= pair.minSim;

      if (passed) {
        console.log(chalk.green(`PASS`) + chalk.gray(` (${(similarity * 100).toFixed(1)}% >= ${pair.minSim * 100}%, ${elapsed.toFixed(0)}ms)`));
      } else {
        console.log(chalk.red(`FAIL`) + chalk.gray(` (${(similarity * 100).toFixed(1)}% < ${pair.minSim * 100}%)`));
      }

      results.push({
        test: pair.name,
        category: 'Embeddings',
        passed,
        score: similarity,
        details: `Similarity: ${(similarity * 100).toFixed(1)}% (min: ${pair.minSim * 100}%)`,
        timeMs: elapsed,
      });

    } catch (error) {
      console.log(chalk.red(`ERROR`));
      results.push({
        test: pair.name,
        category: 'Embeddings',
        passed: false,
        details: `Error: ${error instanceof Error ? error.message : error}`,
        timeMs: performance.now() - start,
      });
    }
  }

  // Test 2: Dissimilar sentences should have lower similarity
  const dissimilarPairs = [
    {
      name: 'Topic Distinction',
      a: 'The Supreme Court issued a landmark ruling on civil rights.',
      b: 'The recipe calls for two cups of flour and one egg.',
      maxSim: 0.5,
    },
    {
      name: 'Domain Separation',
      a: 'Contract breach requires proof of damages.',
      b: 'The weather forecast predicts rain tomorrow.',
      maxSim: 0.4,
    },
  ];

  for (const pair of dissimilarPairs) {
    const start = performance.now();
    process.stdout.write(chalk.gray(`    ${pair.name}: `));

    try {
      const [embA, embB] = await Promise.all([
        getEmbedding(pair.a, modelId),
        getEmbedding(pair.b, modelId),
      ]);
      const elapsed = performance.now() - start;

      const similarity = cosineSimilarity(embA, embB);
      const passed = similarity <= pair.maxSim;

      if (passed) {
        console.log(chalk.green(`PASS`) + chalk.gray(` (${(similarity * 100).toFixed(1)}% <= ${pair.maxSim * 100}%, ${elapsed.toFixed(0)}ms)`));
      } else {
        console.log(chalk.red(`FAIL`) + chalk.gray(` (${(similarity * 100).toFixed(1)}% > ${pair.maxSim * 100}%)`));
      }

      results.push({
        test: pair.name,
        category: 'Embeddings',
        passed,
        score: similarity,
        details: `Similarity: ${(similarity * 100).toFixed(1)}% (max: ${pair.maxSim * 100}%)`,
        timeMs: elapsed,
      });

    } catch (error) {
      console.log(chalk.red(`ERROR`));
      results.push({
        test: pair.name,
        category: 'Embeddings',
        passed: false,
        details: `Error: ${error instanceof Error ? error.message : error}`,
        timeMs: performance.now() - start,
      });
    }
  }

  return results;
}

// ============================================================
// Search Verification
// ============================================================

async function verifySearch(): Promise<VerificationResult[]> {
  const results: VerificationResult[] = [];

  const modelId = await getModelId(EMBEDDINGS_URL);
  if (!modelId) {
    return [{
      test: 'Search Service',
      category: 'Search',
      passed: false,
      details: 'Embeddings service not available',
      timeMs: 0,
    }];
  }

  console.log(chalk.white.bold('\n  Search Ranking Verification Tests\n'));
  console.log(chalk.gray(`  Model: ${modelId}\n`));

  // Document corpus
  const documents = [
    { id: 1, text: 'The court found the defendant liable for breach of contract and awarded damages.' },
    { id: 2, text: 'Patent infringement cases require proof of unauthorized use of protected inventions.' },
    { id: 3, text: 'Criminal law distinguishes between felonies and misdemeanors based on severity.' },
    { id: 4, text: 'The merger was approved after antitrust review by the Federal Trade Commission.' },
    { id: 5, text: 'Employment discrimination claims must be filed within statutory time limits.' },
  ];

  // Search test cases
  const searchTests = [
    {
      name: 'Contract Query',
      query: 'breach of contract damages',
      expectedTop: 1,
    },
    {
      name: 'Patent Query',
      query: 'patent infringement intellectual property',
      expectedTop: 2,
    },
    {
      name: 'Criminal Law Query',
      query: 'felony misdemeanor criminal offense',
      expectedTop: 3,
    },
    {
      name: 'Antitrust Query',
      query: 'merger acquisition FTC approval',
      expectedTop: 4,
    },
  ];

  // Pre-compute document embeddings
  process.stdout.write(chalk.gray(`    Indexing ${documents.length} documents... `));
  const indexStart = performance.now();

  const docEmbeddings: Array<{ id: number; text: string; embedding: number[] }> = [];
  for (const doc of documents) {
    const embedding = await getEmbedding(doc.text, modelId);
    docEmbeddings.push({ ...doc, embedding });
  }
  console.log(chalk.green(`done`) + chalk.gray(` (${(performance.now() - indexStart).toFixed(0)}ms)\n`));

  // Run search tests
  for (const test of searchTests) {
    const start = performance.now();
    process.stdout.write(chalk.gray(`    ${test.name}: `));

    try {
      const queryEmbedding = await getEmbedding(test.query, modelId);
      const elapsed = performance.now() - start;

      // Rank documents by similarity
      const ranked = docEmbeddings
        .map(doc => ({
          id: doc.id,
          score: cosineSimilarity(queryEmbedding, doc.embedding),
        }))
        .sort((a, b) => b.score - a.score);

      const topResult = ranked[0];
      const passed = topResult.id === test.expectedTop;

      if (passed) {
        console.log(chalk.green(`PASS`) + chalk.gray(` (doc #${topResult.id} at ${(topResult.score * 100).toFixed(1)}%, ${elapsed.toFixed(0)}ms)`));
      } else {
        console.log(chalk.red(`FAIL`) + chalk.gray(` (expected #${test.expectedTop}, got #${topResult.id})`));
        console.log(chalk.gray(`      Ranking: ${ranked.map(r => `#${r.id}:${(r.score * 100).toFixed(0)}%`).join(' > ')}`));
      }

      results.push({
        test: test.name,
        category: 'Search',
        passed,
        score: topResult.score,
        expected: `Document #${test.expectedTop}`,
        actual: `Document #${topResult.id}`,
        details: `Top result: #${topResult.id} (${(topResult.score * 100).toFixed(1)}%)`,
        timeMs: elapsed,
      });

    } catch (error) {
      console.log(chalk.red(`ERROR`));
      results.push({
        test: test.name,
        category: 'Search',
        passed: false,
        details: `Error: ${error instanceof Error ? error.message : error}`,
        timeMs: performance.now() - start,
      });
    }
  }

  return results;
}

// ============================================================
// Main
// ============================================================

async function main() {
  const args = process.argv.slice(2);

  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('         OCR & Embedding Verification Demo'));
  console.log(chalk.cyan.bold('='.repeat(70)));

  const runOcr = args.length === 0 || args.includes('--ocr') || args.includes('--all');
  const runEmb = args.length === 0 || args.includes('--embedding') || args.includes('--all');
  const runSearch = args.length === 0 || args.includes('--search') || args.includes('--all');

  const allResults: VerificationResult[] = [];

  if (runOcr) {
    const ocrResults = await verifyOcr();
    allResults.push(...ocrResults);
  }

  if (runEmb) {
    const embResults = await verifyEmbeddings();
    allResults.push(...embResults);
  }

  if (runSearch) {
    const searchResults = await verifySearch();
    allResults.push(...searchResults);
  }

  // Summary
  console.log(chalk.cyan.bold('\n' + '='.repeat(70)));
  console.log(chalk.cyan.bold('                    Verification Summary'));
  console.log(chalk.cyan.bold('='.repeat(70) + '\n'));

  const categories = [...new Set(allResults.map(r => r.category))];

  for (const cat of categories) {
    const catResults = allResults.filter(r => r.category === cat);
    const passed = catResults.filter(r => r.passed).length;
    const total = catResults.length;
    const avgTime = catResults.reduce((sum, r) => sum + r.timeMs, 0) / total;

    const status = passed === total
      ? chalk.green.bold('PASS')
      : passed > 0
        ? chalk.yellow.bold('PARTIAL')
        : chalk.red.bold('FAIL');

    console.log(`  ${cat.padEnd(15)} ${status} ${chalk.gray(`(${passed}/${total} tests, avg ${avgTime.toFixed(0)}ms)`)}`);

    for (const r of catResults) {
      const icon = r.passed ? chalk.green('  ✓') : chalk.red('  ✗');
      console.log(`${icon} ${r.test}: ${chalk.dim(r.details)}`);
    }
    console.log('');
  }

  const totalPassed = allResults.filter(r => r.passed).length;
  const totalTests = allResults.length;
  const totalTime = allResults.reduce((sum, r) => sum + r.timeMs, 0);

  console.log(chalk.white.bold('  Overall:'));
  console.log(chalk.gray(`    Tests: ${totalPassed}/${totalTests} passed`));
  console.log(chalk.gray(`    Time:  ${(totalTime / 1000).toFixed(1)}s`));

  console.log(chalk.cyan.bold('\n' + '='.repeat(70) + '\n'));

  process.exit(totalPassed === totalTests ? 0 : 1);
}

main().catch(error => {
  console.error(chalk.red(`\nFatal error: ${error.message}`));
  process.exit(1);
});
