import { Command } from 'commander';
import { glob } from 'glob';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { waitForServices, extractText, getDocumentEmbedding, getQueryEmbedding, cosineSimilarity, saveOcrMarkdown } from './api.js';
import {
    getSystemInfo,
    computeFileHash,
    generateSessionId,
    calculateSessionStats,
    saveSession,
    formatDuration,
    formatBytes,
    formatNumber,
    estimatePageCount,
    estimateTokens
} from './benchmark-utils.js';
import type { FileMetrics, BenchmarkSession, SearchMetrics } from './types.js';

const program = new Command();
const DB_FILE = 'embeddings.json';
const OCR_DIR = 'ocr';
const DEFAULT_PDF_COUNT = 1;  // Default to 1 for quick first-run experience
const DEFAULT_SEARCH_LIMIT = 20;  // Default number of search results to return

interface IndexedDoc {
    id: string;
    text: string;
    embedding: number[];
    filePath?: string;
    fileSize?: number;
    charCount?: number;
    fileHash?: string;
}

function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function createSnippet(text: string, query: string, snippetLength = 300): string {
    const queryWords = query.toLowerCase().split(/\s+/).filter(Boolean);
    const lowerText = text.toLowerCase();

    let firstMatchIndex = -1;
    for (const word of queryWords) {
        const index = lowerText.indexOf(word);
        if (index !== -1) {
            firstMatchIndex = index;
            break;
        }
    }

    if (firstMatchIndex === -1) {
        return text.length > snippetLength ? text.substring(0, snippetLength) + '...' : text;
    }

    const start = Math.max(0, firstMatchIndex - Math.round(snippetLength / 3));
    let snippet = text.substring(start);

    // Highlight matched words with bold yellow (visible on most terminals)
    for (const word of queryWords) {
        const regex = new RegExp(`(${escapeRegExp(word)})`, 'gi');
        snippet = snippet.replace(regex, chalk.yellow.bold('$1'));
    }

    if (snippet.length > snippetLength) {
        snippet = snippet.substring(0, snippetLength) + '...';
    }

    if (start > 0) {
        snippet = '...' + snippet;
    }

    return snippet.replace(/\n/g, ' ').replace(/\s+/g, ' ');
}

function isLikelyGarbage(text: string, threshold = 0.5): boolean {
    if (!text) return false;
    let nonAscii = 0;
    for (let i = 0; i < text.length; i++) {
        if (text.charCodeAt(i) > 255) {
            nonAscii++;
        }
    }
    return (nonAscii / text.length) > threshold;
}

function getRandomFiles(files: string[], count: number): string[] {
    if (count >= files.length) return files;
    const shuffled = [...files].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
}

function getSimilarityLabel(score: number, max: number, min: number, avg: number): string {
    if (Math.abs(score - max) < 0.0001) return '(max)';
    if (Math.abs(score - min) < 0.0001) return '(min)';
    if (Math.abs(score - avg) < 0.01) return '(avg)';
    return '';
}

function printSimilarityExplanation(): void {
    console.log('\n  [Similarity Score Guide]');
    console.log('  Similarity ranges from 0.0 to 1.0 (higher = more relevant)');
    console.log('  > 0.7: Strong match    0.5-0.7: Moderate    < 0.5: Weak');
    console.log('  Distance = 1 - Similarity (lower distance = closer match)');
}

function printBenchmarkStats(session: BenchmarkSession): void {
    const { stats, system, files, config } = session;
    const ocrPct = stats.totalDurationMs > 0 ? ((stats.totalOcrTimeMs / stats.totalDurationMs) * 100).toFixed(1) : '0';
    const embedPct = stats.totalDurationMs > 0 ? ((stats.totalEmbedTimeMs / stats.totalDurationMs) * 100).toFixed(1) : '0';

    console.log(chalk.bold('\n========================================'));
    console.log(chalk.bold('  Benchmark Statistics'));
    console.log(chalk.bold('========================================'));

    // System & Sample Info - Two column table
    console.log(chalk.bold('\n[System]                              [Sample]'));
    console.log(`  Platform: ${chalk.bold((system.platform + ' ' + system.arch).padEnd(20))}    Files: ${chalk.bold(stats.totalFiles)}`);
    console.log(`  CPU: ${chalk.bold(system.cpuModel.substring(0, 24).padEnd(24))}    Size: ${chalk.bold(stats.totalSizeMB.toFixed(2) + ' MB')}`);
    console.log(`  Cores: ${chalk.bold(String(system.cpuCores).padEnd(22))}    Pages: ${chalk.bold(formatNumber(stats.totalPages))} (est)`);
    console.log(`  Memory: ${chalk.bold((system.totalMemoryGB + ' GB').padEnd(21))}    Chars: ${chalk.bold(formatNumber(stats.totalChars))}`);
    console.log(`  GPU: ${chalk.bold((system.gpuAvailable ? 'Yes' : 'No').padEnd(24))}    Chunks: ${chalk.bold(formatNumber(stats.totalChunks))}`);

    // Search Query
    if (config?.searchQuery) {
        console.log(chalk.bold('\n[Search]'));
        console.log(`  Query: ${chalk.bold.yellow('"' + config.searchQuery + '"')}`);
    }

    // Timing - Horizontal table
    console.log(chalk.bold('\n[Timing]'));
    console.log(`  Total: ${chalk.bold(formatDuration(stats.totalDurationMs).padEnd(12))}  OCR: ${chalk.bold(formatDuration(stats.totalOcrTimeMs))} (${ocrPct}%)    Embed: ${chalk.bold(formatDuration(stats.totalEmbedTimeMs))} (${embedPct}%)`);

    // Performance - Two column table
    console.log(chalk.bold('\n[OCR Performance]                     [Embedding Performance]'));
    console.log(`  Avg: ${chalk.bold(formatDuration(stats.ocrAvgTimeMs).padEnd(14))} per file     Avg: ${chalk.bold(formatDuration(stats.embedAvgTimeMs))} per file`);
    console.log(`  Min: ${chalk.bold(formatDuration(stats.ocrMinTimeMs).padEnd(14))}              Min: ${chalk.bold(formatDuration(stats.embedMinTimeMs))}`);
    console.log(`  Max: ${chalk.bold(formatDuration(stats.ocrMaxTimeMs).padEnd(14))}              Max: ${chalk.bold(formatDuration(stats.embedMaxTimeMs))}`);
    console.log(`  Rate: ${chalk.bold((formatNumber(stats.ocrAvgCharsPerSecond) + ' c/s').padEnd(13))}             Rate: ${chalk.bold(formatNumber(stats.embedAvgCharsPerSecond) + ' c/s')}`);
    console.log(`                                        Tokens: ${chalk.bold(formatNumber(stats.embedAvgTokensPerSecond) + ' t/s')}`);

    // Projections - Horizontal
    console.log(chalk.bold('\n[Projections]'));
    console.log(`  100 chars: ${chalk.bold(formatDuration(stats.estimatedEmbedPer100Chars_ms).padEnd(8))}  1K chars: ${chalk.bold(formatDuration(stats.estimatedEmbedPer1000Chars_ms).padEnd(8))}  100 tokens: ${chalk.bold(formatDuration(stats.estimatedEmbedPer100Tokens_ms))}`);
    console.log(`  1 MB: ${chalk.bold(formatDuration(stats.estimatedTimePer1MB_ms).padEnd(13))}  100 MB: ${chalk.bold(formatDuration(stats.estimatedTimePer100MB_ms).padEnd(10))}  1 GB: ${chalk.bold(formatDuration(stats.estimatedTimePer1GB_ms))}`);

    // Text Comparison Stats (if any files have comparison data)
    const filesWithComparison = files.filter(f => f.rawTextChars !== undefined && f.rawTextChars > 0);
    if (filesWithComparison.length > 0) {
        const avgSimilarity = filesWithComparison.reduce((sum, f) => sum + (f.textSimilarityPercent || 0), 0) / filesWithComparison.length;
        const avgDifference = filesWithComparison.reduce((sum, f) => sum + (f.textDifferencePercent || 0), 0) / filesWithComparison.length;
        const totalRawChars = filesWithComparison.reduce((sum, f) => sum + (f.rawTextChars || 0), 0);
        const totalOcrChars = filesWithComparison.reduce((sum, f) => sum + f.ocrOutputChars, 0);

        console.log(chalk.bold('\n[Text Comparison] (PDF raw text vs OCR)'));
        console.log(`  Files with embedded text: ${chalk.bold(filesWithComparison.length)}/${files.length}`);
        console.log(`  Total raw chars: ${chalk.bold(formatNumber(totalRawChars))}    Total OCR chars: ${chalk.bold(formatNumber(totalOcrChars))}`);
        console.log(`  Avg similarity: ${chalk.bold(avgSimilarity.toFixed(1) + '%')}    Avg difference: ${chalk.bold(avgDifference.toFixed(1) + '%')}`);
    }

    // Per-File Details (if not too many)
    if (files.length > 0 && files.length <= 10) {
        const hasComparison = files.some(f => f.rawTextChars !== undefined);

        console.log(chalk.bold('\n[Files]'));
        if (hasComparison) {
            console.log('  Hash             Size       Pages   Raw        OCR        Similarity  Embed      Chunks');
            console.log('  ---------------  ---------  ------  ---------  ---------  ----------  ---------  ------');
        } else {
            console.log('  Hash             Size       Pages   Chars      OCR        Embed      Chunks');
            console.log('  ---------------  ---------  ------  ---------  ---------  ---------  ------');
        }

        for (const f of files) {
            const hash = f.fileHash.substring(0, 15);
            const size = formatBytes(f.fileSizeBytes).padEnd(9);
            const pages = String(f.pageCount).padStart(6);
            const ocrTime = formatDuration(f.ocrDurationMs).padStart(9);
            const embedTime = formatDuration(f.embedDurationMs).padStart(9);
            const chunks = String(f.chunkCount).padStart(6);

            if (hasComparison) {
                const rawChars = f.rawTextChars !== undefined ? formatNumber(f.rawTextChars).padStart(9) : '       -'.padStart(9);
                const ocrChars = formatNumber(f.ocrOutputChars).padStart(9);
                const similarity = f.textSimilarityPercent !== undefined ? (f.textSimilarityPercent.toFixed(1) + '%').padStart(10) : '        -'.padStart(10);
                console.log(`  ${chalk.bold(hash)}  ${chalk.bold(size)}  ${chalk.bold(pages)}  ${chalk.bold(rawChars)}  ${chalk.bold(ocrChars)}  ${chalk.bold(similarity)}  ${chalk.bold(embedTime)}  ${chalk.bold(chunks)}`);
            } else {
                const chars = formatNumber(f.ocrOutputChars).padStart(9);
                console.log(`  ${chalk.bold(hash)}  ${chalk.bold(size)}  ${chalk.bold(pages)}  ${chalk.bold(chars)}  ${chalk.bold(ocrTime)}  ${chalk.bold(embedTime)}  ${chalk.bold(chunks)}`);
            }
        }
    }

    // Session Info - Compact
    console.log(chalk.bold('\n[Session]'));
    console.log(`  ID: ${chalk.bold(session.sessionId)}    Time: ${session.timestampIso}    Log: logs/${session.sessionId}.json`);
}

async function indexFiles(
    filePathOrDir: string,
    options: { pdfCount?: number; collectMetrics?: boolean } = {}
): Promise<{ db: IndexedDoc[]; fileMetrics: FileMetrics[] }> {
    const fileMetrics: FileMetrics[] = [];
    const mistralApiKey = process.env.MISTRAL_OCR_API_KEY;

    if (!mistralApiKey) {
        console.error(chalk.red("[ERROR] MISTRAL_OCR_API_KEY environment variable not set."));
        return { db: [], fileMetrics: [] };
    }

    await waitForServices();
    console.log(chalk.blue(`\n[Index] Source: ${filePathOrDir}`));

    if (!fs.existsSync(OCR_DIR)) {
        fs.mkdirSync(OCR_DIR, { recursive: true });
    }

    let files: string[];
    if (fs.statSync(filePathOrDir).isDirectory()) {
        files = await glob(`${filePathOrDir}/**/*.{pdf,png,jpg,jpeg,gif,bmp,tiff,docx,pptx,txt,md}`);
    } else {
        files = [filePathOrDir];
    }

    const pdfCount = options.pdfCount;
    if (pdfCount !== undefined && pdfCount > 0 && files.length > pdfCount) {
        console.log(chalk.yellow(`[Limit] Selecting ${pdfCount} random files from ${files.length} available`));
        files = getRandomFiles(files, pdfCount);
    }

    const db: IndexedDoc[] = [];
    console.log(chalk.blue(`[Files] Found ${files.length} file(s) to process.`));

    for (const [index, file] of files.entries()) {
        const fileSize = fs.statSync(file).size;
        const filePath = path.resolve(file);
        const fileHash = computeFileHash(file);
        const estimatedPages = estimatePageCount(fileSize);

        console.log(chalk.cyan(`\n[${index + 1}/${files.length}] ${path.basename(file)}`));
        console.log(chalk.gray(`  Hash: ${fileHash}`));
        console.log(chalk.gray(`  Size: ${formatBytes(fileSize)} (~${estimatedPages} pages)`));

        // Initialize metrics
        const metrics: FileMetrics = {
            fileHash,
            fileSizeBytes: fileSize,
            fileSizeMB: fileSize / (1024 * 1024),
            pageCount: estimatedPages,
            estimatedChars: 0,
            ocrStartTime: 0,
            ocrEndTime: 0,
            ocrDurationMs: 0,
            ocrOutputChars: 0,
            ocrOutputSizeBytes: 0,
            embedStartTime: 0,
            embedEndTime: 0,
            embedDurationMs: 0,
            chunkCount: 0,
            tokensEstimate: 0,
            ocrCharsPerSecond: 0,
            embedCharsPerSecond: 0,
            embedTokensPerSecond: 0
        };

        // OCR Phase
        const ocrSpinner = ora('OCR with Mistral...').start();
        metrics.ocrStartTime = performance.now();
        const extractResult = await extractText(file, mistralApiKey);
        metrics.ocrEndTime = performance.now();
        metrics.ocrDurationMs = extractResult.duration;

        const text = extractResult.text;
        if (!text) {
            ocrSpinner.fail("[FAIL] OCR failed (no text extracted).");
            continue;
        }

        metrics.ocrOutputChars = text.length;
        metrics.ocrOutputSizeBytes = Buffer.byteLength(text, 'utf-8');
        metrics.ocrCharsPerSecond = extractResult.duration > 0 ? (text.length / (extractResult.duration / 1000)) : 0;

        // Add text comparison data if available
        if (extractResult.comparison) {
            metrics.rawTextChars = extractResult.comparison.rawTextChars;
            metrics.textSimilarityPercent = extractResult.comparison.similarityPercent;
            metrics.textDifferencePercent = extractResult.comparison.differencePercent;
        }

        // Update page count if we got it from PDF extraction
        if (extractResult.pageCount && extractResult.pageCount > 0) {
            metrics.pageCount = extractResult.pageCount;
        }

        ocrSpinner.succeed(`[OK] OCR done in ${formatDuration(extractResult.duration)}. ${formatNumber(text.length)} chars.`);

        // Show comparison stats if raw text was available
        if (extractResult.comparison?.rawTextAvailable) {
            const comp = extractResult.comparison;
            console.log(chalk.gray(`  [Compare] Raw: ${formatNumber(comp.rawTextChars)} chars    OCR: ${formatNumber(comp.ocrTextChars)} chars    Similarity: ${comp.similarityPercent.toFixed(1)}%`));
        }

        // Save OCR output
        const ocrFilePath = await saveOcrMarkdown(file, text);
        console.log(chalk.gray(`  OCR saved: ${ocrFilePath}`));

        if (isLikelyGarbage(text)) {
            console.log(chalk.red("  [SKIP] Text appears to be garbled/corrupt."));
            continue;
        }

        const MAX_TEXT_LENGTH = 10000000;
        if (text.length > MAX_TEXT_LENGTH) {
            console.log(chalk.red(`  [SKIP] Text too long (${formatNumber(text.length)} > ${formatNumber(MAX_TEXT_LENGTH)})`));
            continue;
        }

        // Embedding Phase
        const embedSpinner = ora('Embedding with Inception...').start();
        metrics.embedStartTime = performance.now();

        try {
            const embeddingResponse = await getDocumentEmbedding(text);
            metrics.embedEndTime = performance.now();
            metrics.embedDurationMs = metrics.embedEndTime - metrics.embedStartTime;

            if (embeddingResponse && embeddingResponse.embeddings && Array.isArray(embeddingResponse.embeddings)) {
                const numChunks = embeddingResponse.embeddings.length;
                metrics.chunkCount = numChunks;
                metrics.tokensEstimate = estimateTokens(text.length);
                metrics.embedCharsPerSecond = metrics.embedDurationMs > 0
                    ? (text.length / (metrics.embedDurationMs / 1000))
                    : 0;
                metrics.embedTokensPerSecond = metrics.embedDurationMs > 0
                    ? (metrics.tokensEstimate / (metrics.embedDurationMs / 1000))
                    : 0;

                embedSpinner.succeed(`[OK] Embedding done in ${formatDuration(metrics.embedDurationMs)}. ${numChunks} chunks.`);

                for (const item of embeddingResponse.embeddings) {
                    if (item.embedding && Array.isArray(item.embedding)) {
                        db.push({
                            id: `${path.basename(file)}#${item.chunk_number}`,
                            text: item.chunk,
                            embedding: item.embedding,
                            filePath: filePath,
                            fileSize: fileSize,
                            charCount: item.chunk.length,
                            fileHash: fileHash
                        });
                    }
                }

                fileMetrics.push(metrics);
            } else {
                embedSpinner.fail("[FAIL] Embedding failed (unexpected response).");
            }
        } catch (e) {
            metrics.embedEndTime = performance.now();
            metrics.embedDurationMs = metrics.embedEndTime - metrics.embedStartTime;
            embedSpinner.fail(`[FAIL] Embedding failed in ${formatDuration(metrics.embedDurationMs)}.`);
        }
    }

    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
    console.log(chalk.green(`\n[Index Complete] ${db.length} chunks saved to ${DB_FILE}`));

    return { db, fileMetrics };
}

async function search(query: string, limit: number = DEFAULT_SEARCH_LIMIT): Promise<SearchMetrics | undefined> {
    const searchStartTime = performance.now();

    if (!fs.existsSync(DB_FILE)) {
        console.log(chalk.red(`[ERROR] No index found. Run 'index' command first.`));
        return undefined;
    }

    const db: IndexedDoc[] = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));

    await waitForServices();
    console.log(chalk.blue(`\n[Search] Query: "${query}"`));

    const queryEmbStartTime = performance.now();
    const queryEmbResponse = await getQueryEmbedding(query);
    const queryEmbTime = performance.now() - queryEmbStartTime;

    let queryEmb: number[] = [];
    if (Array.isArray(queryEmbResponse)) {
        queryEmb = queryEmbResponse;
    } else if (queryEmbResponse.embedding) {
        queryEmb = queryEmbResponse.embedding;
    }

    if (queryEmb.length === 0) {
        console.error(chalk.red("[ERROR] Failed to get query embedding"));
        return undefined;
    }

    const similarityStartTime = performance.now();
    const results = db.map(doc => ({
        doc,
        score: cosineSimilarity(queryEmb, doc.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
    const similarityTime = performance.now() - similarityStartTime;

    const totalSearchTime = performance.now() - searchStartTime;

    const searchMetrics: SearchMetrics = {
        query,
        queryEmbedTimeMs: queryEmbTime,
        similaritySearchTimeMs: similarityTime,
        totalSearchTimeMs: totalSearchTime,
        chunksSearched: db.length,
        resultsReturned: results.length,
        throughputChunksPerSec: similarityTime > 0 ? (db.length / (similarityTime / 1000)) : 0
    };

    if (results.length === 0) {
        console.log(chalk.yellow("[INFO] No results found."));
        return searchMetrics;
    }

    const scores = results.map(r => r.score);
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    printSimilarityExplanation();

    console.log(chalk.bold(`\n[Search Statistics] (Top ${results.length} results)`));
    console.log(`  Max: ${maxScore.toFixed(4)} (best)    Min: ${minScore.toFixed(4)} (weakest)    Avg: ${avgScore.toFixed(4)}`);

    console.log(chalk.bold(`\n[Top Relevant Matches]`));

    results.forEach((res, i) => {
        const [filename, chunkNum] = res.doc.id.split('#');
        const snippet = createSnippet(res.doc.text, query);
        const similarityLabel = getSimilarityLabel(res.score, maxScore, minScore, avgScore);

        console.log(`\n--- Match ${i+1} ---`);
        console.log(`  File: ${filename}    Chunk: #${chunkNum || 'N/A'}    Similarity: ${res.score.toFixed(4)} ${similarityLabel}`);
        if (res.doc.fileSize && res.doc.charCount) {
            console.log(`  Size: ${formatBytes(res.doc.fileSize)}    Chars: ${formatNumber(res.doc.charCount)}`);
        }
        console.log(`  Snippet: "${snippet}"`);
    });

    console.log(chalk.bold(`\n[Search Timing]`));
    console.log(`  Query embed: ${formatDuration(queryEmbTime)}    Search: ${formatDuration(similarityTime)} (${db.length} chunks)    Total: ${formatDuration(totalSearchTime)}`);
    console.log(`  Throughput: ${formatNumber(Math.round(searchMetrics.throughputChunksPerSec))} chunks/sec`);

    return searchMetrics;
}

program
    .name('inception-client')
    .description('CLI to demo Inception with Mistral OCR for document indexing and semantic search')
    .version('1.0.0');

program.command('index')
    .argument('<path>', 'File or directory to index')
    .option('--pdf-count <n>', 'Number of PDFs to index (random selection if more available)', parseInt)
    .description('Index document(s) and generate embeddings')
    .action(async (filePath: string, options: { pdfCount?: number }) => {
        await indexFiles(filePath, options);
    });

program.command('search')
    .argument('<query>', 'Search query')
    .option('--limit <n>', `Number of results to return (default: ${DEFAULT_SEARCH_LIMIT})`, parseInt)
    .description('Search indexed documents using semantic search')
    .action(async (query: string, options: { limit?: number }) => {
        await search(query, options.limit || DEFAULT_SEARCH_LIMIT);
    });

program.command('run')
    .argument('<file>', 'File to index')
    .argument('<query>', 'Search query')
    .description('Index a single file and search it')
    .action(async (file: string, query: string) => {
        const sourcePath = path.resolve(file);

        if (!fs.existsSync(sourcePath)) {
            console.error(chalk.red(`[ERROR] File not found: ${file}`));
            return;
        }

        const filesDir = path.join(process.cwd(), 'files');
        if (!fs.existsSync(filesDir)) {
            fs.mkdirSync(filesDir, { recursive: true });
        }

        const fileName = path.basename(file);
        const destPath = path.join(filesDir, fileName);

        if (path.resolve(sourcePath) !== path.resolve(destPath)) {
            console.log(chalk.blue(`[Copy] ${fileName} -> files/`));
            fs.copyFileSync(sourcePath, destPath);
            console.log(chalk.green(`[OK] File copied\n`));
        }

        await indexFiles(destPath);
        await search(query);
    });

program.command('demo')
    .argument('[file]', 'Optional: File to add to demo')
    .argument('[query]', 'Optional: Search query (default: "fraud")')
    .option('--pdf-count <n>', `Number of PDFs to index (default: ${DEFAULT_PDF_COUNT}, use 0 for all)`, parseInt)
    .option('--no-save', 'Do not save benchmark session to logs/')
    .description('Run demo: index files and search with benchmark output')
    .action(async (file?: string, query?: string, options?: { pdfCount?: number; save?: boolean }) => {
        const filesDir = path.join(process.cwd(), 'files');
        const pdfCount = options?.pdfCount !== undefined ? options.pdfCount : DEFAULT_PDF_COUNT;
        const shouldSave = options?.save !== false;

        console.log(chalk.blue.bold('\n========================================'));
        console.log(chalk.blue.bold('  Inception Demo - Document Search'));
        console.log(chalk.blue.bold('========================================\n'));

        // Collect system info
        const systemInfo = getSystemInfo();
        const sessionId = generateSessionId();

        console.log(chalk.gray(`[Session] ${sessionId}`));
        console.log(chalk.gray(`[System] ${systemInfo.platform} ${systemInfo.arch} - ${systemInfo.cpuModel}`));
        console.log(chalk.gray(`[Memory] ${systemInfo.totalMemoryGB} GB total`));

        if (file) {
            const sourcePath = path.resolve(file);
            if (!fs.existsSync(sourcePath)) {
                console.error(chalk.red(`[ERROR] File not found: ${file}`));
                return;
            }

            if (!fs.existsSync(filesDir)) {
                fs.mkdirSync(filesDir, { recursive: true });
            }

            const fileName = path.basename(file);
            const destPath = path.join(filesDir, fileName);

            if (path.resolve(sourcePath) !== path.resolve(destPath)) {
                console.log(chalk.blue(`[Copy] Adding ${fileName} to demo files...`));
                fs.copyFileSync(sourcePath, destPath);
            }
        }

        if (!fs.existsSync(filesDir)) {
            console.error(chalk.red(`[ERROR] Files directory not found: ${filesDir}`));
            return;
        }

        console.log(chalk.blue.bold('\n[Phase 1] Indexing Documents'));
        console.log(chalk.gray(`  PDF limit: ${pdfCount === 0 ? 'All files' : pdfCount + ' file(s)'}`));

        const { fileMetrics } = await indexFiles(filesDir, {
            pdfCount: pdfCount === 0 ? undefined : pdfCount,
            collectMetrics: true
        });

        // Search
        const searchQuery = query || "fraud";
        console.log(chalk.blue.bold(`\n[Phase 2] Semantic Search`));
        console.log(chalk.gray(`  Query: "${searchQuery}"`));

        const searchMetrics = await search(searchQuery);

        // Calculate stats and create session
        const stats = calculateSessionStats(fileMetrics);
        const now = new Date();

        const session: BenchmarkSession = {
            sessionId,
            timestamp: sessionId,
            timestampIso: now.toISOString(),
            system: systemInfo,
            config: {
                inceptionUrl: process.env.INCEPTION_URL || 'http://localhost:8005',
                modelName: 'freelawproject/modernbert-embed-base_finetune_512',
                pdfCount: pdfCount === 0 ? fileMetrics.length : pdfCount,
                searchQuery
            },
            files: fileMetrics,
            stats,
            search: searchMetrics
        };

        // Print benchmark stats
        printBenchmarkStats(session);

        // Save session
        if (shouldSave && fileMetrics.length > 0) {
            const savedPath = saveSession(session);
            console.log(chalk.green(`\n[Saved] Benchmark session: ${savedPath}`));
        }

        console.log(chalk.blue.bold('\n========================================'));
        console.log(chalk.blue.bold('  Demo Complete'));
        console.log(chalk.blue.bold('========================================'));
    });

program.command('benchmark')
    .argument('[folder]', 'Folder containing session JSON files', 'logs')
    .description('Analyze benchmark sessions and compare performance')
    .action(async (folder: string) => {
        // Import and run the benchmark analyzer
        const { analyzeBenchmarks } = await import('./benchmark.js');
        analyzeBenchmarks(folder);
    });

program.parse();
