#!/usr/bin/env npx tsx
/**
 * Test script to compare grep/regex text search vs semantic vector search
 * This helps validate and tune the semantic search quality
 */

import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { glob } from 'glob';
import { waitForServices, getQueryEmbedding, cosineSimilarity } from './api.js';

const DB_FILE = 'embeddings.json';
const OCR_DIR = 'ocr';

interface IndexedDoc {
    id: string;
    text: string;
    embedding: number[];
    filePath?: string;
    fileSize?: number;
    charCount?: number;
    fileHash?: string;
}

interface GrepMatch {
    fileHash: string;
    line: number;
    text: string;
}

interface FileMatchSummary {
    fileHash: string;
    grepCount: number;
    semanticRank: number | null;
    semanticScore: number | null;
    semanticChunks: number;
}

function grepInOcrFiles(query: string): Map<string, GrepMatch[]> {
    const results = new Map<string, GrepMatch[]>();
    const pattern = new RegExp(query, 'gi');

    const ocrFiles = fs.readdirSync(OCR_DIR).filter(f => f.endsWith('.ocr.md'));

    for (const file of ocrFiles) {
        const fileHash = file.replace('.ocr.md', '');
        const content = fs.readFileSync(path.join(OCR_DIR, file), 'utf-8');
        const lines = content.split('\n');
        const matches: GrepMatch[] = [];

        for (let i = 0; i < lines.length; i++) {
            if (pattern.test(lines[i])) {
                matches.push({
                    fileHash,
                    line: i + 1,
                    text: lines[i].substring(0, 150)
                });
            }
            // Reset lastIndex for global regex
            pattern.lastIndex = 0;
        }

        if (matches.length > 0) {
            results.set(fileHash, matches);
        }
    }

    return results;
}

async function semanticSearch(query: string, limit: number = 50): Promise<{ doc: IndexedDoc; score: number }[]> {
    if (!fs.existsSync(DB_FILE)) {
        console.error(chalk.red(`[ERROR] No embeddings found. Run 'index' command first.`));
        return [];
    }

    const db: IndexedDoc[] = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));

    await waitForServices();

    const queryEmbResponse = await getQueryEmbedding(query);
    let queryEmb: number[] = [];

    if (Array.isArray(queryEmbResponse)) {
        queryEmb = queryEmbResponse;
    } else if (queryEmbResponse.embedding) {
        queryEmb = queryEmbResponse.embedding;
    }

    if (queryEmb.length === 0) {
        console.error(chalk.red("[ERROR] Failed to get query embedding"));
        return [];
    }

    return db.map(doc => ({
        doc,
        score: cosineSimilarity(queryEmb, doc.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function extractFileHash(docId: string): string {
    // docId format: "filename.pdf#chunkNum"
    const parts = docId.split('#');
    const filename = parts[0];
    return filename.replace('.pdf', '').replace('.ocr.md', '');
}

async function compareSearchMethods(query: string): Promise<void> {
    console.log(chalk.blue.bold('\n========================================'));
    console.log(chalk.blue.bold('  Search Quality Comparison'));
    console.log(chalk.blue.bold('========================================'));
    console.log(chalk.gray(`  Query: "${query}"\n`));

    // Run grep search
    console.log(chalk.bold('[1] Grep/Regex Search'));
    const grepResults = grepInOcrFiles(query);
    const grepFileHashes = [...grepResults.keys()];
    const totalGrepMatches = [...grepResults.values()].reduce((sum, arr) => sum + arr.length, 0);

    console.log(`  Files matched: ${chalk.bold(grepFileHashes.length)}`);
    console.log(`  Total matches: ${chalk.bold(totalGrepMatches)}`);

    // Sort by match count
    const sortedGrepFiles = grepFileHashes.sort((a, b) =>
        (grepResults.get(b)?.length || 0) - (grepResults.get(a)?.length || 0)
    );

    console.log('\n  Files by match count:');
    for (const hash of sortedGrepFiles) {
        const count = grepResults.get(hash)?.length || 0;
        console.log(`    ${hash.substring(0, 16)}: ${chalk.yellow(count)} matches`);
    }

    // Run semantic search
    console.log(chalk.bold('\n[2] Semantic Vector Search'));
    const semanticResults = await semanticSearch(query, 100);

    // Group by file
    const fileScores = new Map<string, { maxScore: number; chunks: number; topRank: number }>();

    for (let i = 0; i < semanticResults.length; i++) {
        const res = semanticResults[i];
        const fileHash = extractFileHash(res.doc.id);

        if (!fileScores.has(fileHash)) {
            fileScores.set(fileHash, { maxScore: res.score, chunks: 1, topRank: i + 1 });
        } else {
            const existing = fileScores.get(fileHash)!;
            existing.chunks++;
            if (res.score > existing.maxScore) {
                existing.maxScore = res.score;
            }
        }
    }

    console.log(`  Unique files in top 100: ${chalk.bold(fileScores.size)}`);

    // Sort by max score
    const sortedSemanticFiles = [...fileScores.entries()].sort((a, b) => b[1].maxScore - a[1].maxScore);

    console.log('\n  Files by semantic score:');
    for (const [hash, data] of sortedSemanticFiles.slice(0, 15)) {
        const scoreColor = data.maxScore >= 0.5 ? chalk.green : data.maxScore >= 0.4 ? chalk.yellow : chalk.red;
        console.log(`    ${hash.substring(0, 16)}: ${scoreColor(data.maxScore.toFixed(4))} (${data.chunks} chunks, rank #${data.topRank})`);
    }

    // Compare coverage
    console.log(chalk.bold('\n[3] Coverage Comparison'));

    const semanticFileHashes = new Set([...fileScores.keys()]);
    const grepFileHashSet = new Set(grepFileHashes);

    // Files found by grep but NOT in semantic top results
    const missedBySemanticFull = [...grepFileHashSet].filter(h => !semanticFileHashes.has(h));

    // Files in semantic but NOT in grep
    const extraSemanticFiles = [...semanticFileHashes].filter(h => !grepFileHashSet.has(h));

    // Files found by both
    const bothFound = [...grepFileHashSet].filter(h => semanticFileHashes.has(h));

    console.log(`  Grep files: ${chalk.bold(grepFileHashes.length)}`);
    console.log(`  Semantic files (top 100): ${chalk.bold(fileScores.size)}`);
    console.log(`  Found by both: ${chalk.green.bold(bothFound.length)}`);
    console.log(`  Grep-only (missed by semantic): ${chalk.red.bold(missedBySemanticFull.length)}`);
    console.log(`  Semantic-only (no exact match): ${chalk.yellow.bold(extraSemanticFiles.length)}`);

    if (missedBySemanticFull.length > 0) {
        console.log(chalk.red('\n  Files with grep matches but low/no semantic score:'));
        for (const hash of missedBySemanticFull) {
            const count = grepResults.get(hash)?.length || 0;
            console.log(`    ${hash.substring(0, 16)}: ${count} grep matches, NOT in top 100 semantic`);
        }
    }

    // Detailed comparison table
    console.log(chalk.bold('\n[4] File-by-File Comparison'));
    console.log('  FileHash           Grep   Semantic   Score    Semantic Rank');
    console.log('  ---------------   -----   --------   ------   -------------');

    // Combine all files
    const allFiles = new Set([...grepFileHashes, ...semanticFileHashes]);
    const comparisonData: FileMatchSummary[] = [];

    for (const hash of allFiles) {
        const grepCount = grepResults.get(hash)?.length || 0;
        const semanticData = fileScores.get(hash);

        comparisonData.push({
            fileHash: hash,
            grepCount,
            semanticRank: semanticData?.topRank || null,
            semanticScore: semanticData?.maxScore || null,
            semanticChunks: semanticData?.chunks || 0
        });
    }

    // Sort by grep count (descending)
    comparisonData.sort((a, b) => b.grepCount - a.grepCount);

    for (const data of comparisonData) {
        const hash = data.fileHash.substring(0, 16);
        const grep = data.grepCount > 0 ? chalk.green(String(data.grepCount).padStart(5)) : chalk.gray('    0');
        const semantic = data.semanticChunks > 0 ? chalk.green(String(data.semanticChunks).padStart(8)) : chalk.gray('       0');
        const score = data.semanticScore ? data.semanticScore.toFixed(4).padStart(8) : chalk.gray('       -');
        const rank = data.semanticRank ? String(data.semanticRank).padStart(13) : chalk.gray('            -');

        console.log(`  ${hash}   ${grep}   ${semantic}   ${score}   ${rank}`);
    }

    // Summary
    console.log(chalk.bold('\n[5] Summary'));
    const coverage = bothFound.length / grepFileHashes.length * 100;
    const coverageColor = coverage >= 80 ? chalk.green : coverage >= 50 ? chalk.yellow : chalk.red;

    console.log(`  Semantic coverage of grep files: ${coverageColor(coverage.toFixed(1) + '%')}`);
    console.log(`  Precision (semantic finds relevant): ${chalk.bold((bothFound.length / fileScores.size * 100).toFixed(1) + '%')}`);

    if (coverage < 80) {
        console.log(chalk.yellow('\n  Note: Low coverage may indicate:'));
        console.log(chalk.yellow('    - Embeddings need re-indexing'));
        console.log(chalk.yellow('    - Semantic model captures different concepts'));
        console.log(chalk.yellow('    - Some files have the word but not the semantic meaning'));
    }

    console.log(chalk.blue.bold('\n========================================\n'));
}

// Run the comparison
const query = process.argv[2] || 'fraud';
compareSearchMethods(query).catch(console.error);
