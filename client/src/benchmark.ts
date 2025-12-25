#!/usr/bin/env bun
// Benchmark Analyzer for Inception Demo
// Analyzes and compares benchmark sessions from multiple runs/machines

import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import type { BenchmarkSession, BenchmarkComparison } from './types.js';
import { formatDuration, formatBytes, formatNumber } from './benchmark-utils.js';

interface SessionSummary {
    sessionId: string;
    timestamp: string;
    platform: string;
    cpuModel: string;
    cpuCores: number;
    memoryGB: number;
    gpuAvailable: boolean;
    gpuInfo?: string;
    filesProcessed: number;
    totalSizeMB: number;
    totalChars: number;
    totalChunks: number;
    totalTimeMs: number;
    ocrTimeMs: number;
    embedTimeMs: number;
    ocrAvgCharsPerSec: number;
    embedAvgCharsPerSec: number;
    embedAvgTokensPerSec: number;
    timePer1MB_ms: number;
}

function loadSessions(folder: string): BenchmarkSession[] {
    if (!fs.existsSync(folder)) {
        console.error(chalk.red(`[ERROR] Folder not found: ${folder}`));
        return [];
    }

    const files = fs.readdirSync(folder).filter(f => f.endsWith('.json'));
    const sessions: BenchmarkSession[] = [];

    for (const file of files) {
        try {
            const content = fs.readFileSync(path.join(folder, file), 'utf-8');
            const session = JSON.parse(content) as BenchmarkSession;
            if (session.sessionId && session.stats) {
                sessions.push(session);
            }
        } catch (e) {
            console.warn(chalk.yellow(`[WARN] Could not parse ${file}`));
        }
    }

    return sessions.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function createSummary(session: BenchmarkSession): SessionSummary {
    return {
        sessionId: session.sessionId,
        timestamp: session.timestampIso,
        platform: `${session.system.platform} ${session.system.arch}`,
        cpuModel: session.system.cpuModel,
        cpuCores: session.system.cpuCores,
        memoryGB: session.system.totalMemoryGB,
        gpuAvailable: session.system.gpuAvailable,
        gpuInfo: session.system.gpuInfo,
        filesProcessed: session.stats.totalFiles,
        totalSizeMB: session.stats.totalSizeMB,
        totalChars: session.stats.totalChars,
        totalChunks: session.stats.totalChunks,
        totalTimeMs: session.stats.totalDurationMs,
        ocrTimeMs: session.stats.totalOcrTimeMs,
        embedTimeMs: session.stats.totalEmbedTimeMs,
        ocrAvgCharsPerSec: session.stats.ocrAvgCharsPerSecond,
        embedAvgCharsPerSec: session.stats.embedAvgCharsPerSecond,
        embedAvgTokensPerSec: session.stats.embedAvgTokensPerSecond,
        timePer1MB_ms: session.stats.estimatedTimePer1MB_ms
    };
}

function printSessionComparison(sessions: BenchmarkSession[]): void {
    if (sessions.length === 0) {
        console.log(chalk.yellow('[INFO] No sessions to analyze.'));
        return;
    }

    const summaries = sessions.map(createSummary);

    console.log(chalk.blue.bold('\n========================================'));
    console.log(chalk.blue.bold('  Benchmark Analysis Report'));
    console.log(chalk.blue.bold('========================================'));

    // Overview
    console.log(chalk.cyan('\n[Sessions Overview]'));
    console.log(`  Total Sessions:     ${sessions.length}`);
    console.log(`  Date Range:         ${summaries[0].timestamp.split('T')[0]} to ${summaries[summaries.length - 1].timestamp.split('T')[0]}`);

    // Unique systems
    const uniqueSystems = new Set(summaries.map(s => `${s.platform}|${s.cpuModel}`));
    console.log(`  Unique Systems:     ${uniqueSystems.size}`);

    // Session Details Table
    console.log(chalk.cyan('\n[Session Details]'));
    console.log(chalk.gray('  Session ID         Platform          CPU Cores  Memory   GPU      Files  Size MB   Total Time'));
    console.log(chalk.gray('  -----------------  ----------------  ---------  -------  -------  -----  --------  ----------'));

    for (const s of summaries) {
        const sessionId = s.sessionId.padEnd(17);
        const platform = s.platform.substring(0, 16).padEnd(16);
        const cores = String(s.cpuCores).padStart(9);
        const memory = `${s.memoryGB} GB`.padStart(7);
        const gpu = (s.gpuAvailable ? 'Yes' : 'No').padStart(7);
        const files = String(s.filesProcessed).padStart(5);
        const size = s.totalSizeMB.toFixed(1).padStart(8);
        const time = formatDuration(s.totalTimeMs).padStart(10);

        console.log(`  ${sessionId}  ${platform}  ${cores}  ${memory}  ${gpu}  ${files}  ${size}  ${time}`);
    }

    // Performance Comparison
    console.log(chalk.cyan('\n[Performance Comparison]'));
    console.log(chalk.gray('  Session ID         OCR chars/s   Embed chars/s  Embed tok/s   Time/MB'));
    console.log(chalk.gray('  -----------------  ------------  -------------  ------------  ----------'));

    for (const s of summaries) {
        const sessionId = s.sessionId.padEnd(17);
        const ocrSpeed = formatNumber(s.ocrAvgCharsPerSec).padStart(12);
        const embedSpeed = formatNumber(s.embedAvgCharsPerSec).padStart(13);
        const tokenSpeed = formatNumber(s.embedAvgTokensPerSec).padStart(12);
        const timePerMB = formatDuration(s.timePer1MB_ms).padStart(10);

        console.log(`  ${sessionId}  ${ocrSpeed}  ${embedSpeed}  ${tokenSpeed}  ${timePerMB}`);
    }

    // Find best performers
    if (summaries.length > 1) {
        console.log(chalk.cyan('\n[Performance Rankings]'));

        // Best OCR throughput
        const bestOcr = summaries.reduce((best, s) =>
            s.ocrAvgCharsPerSec > best.ocrAvgCharsPerSec ? s : best
        );
        console.log(`  Best OCR Throughput:      ${bestOcr.sessionId} (${formatNumber(bestOcr.ocrAvgCharsPerSec)} chars/s)`);

        // Best Embedding throughput
        const bestEmbed = summaries.reduce((best, s) =>
            s.embedAvgCharsPerSec > best.embedAvgCharsPerSec ? s : best
        );
        console.log(`  Best Embed Throughput:    ${bestEmbed.sessionId} (${formatNumber(bestEmbed.embedAvgCharsPerSec)} chars/s)`);

        // Fastest per MB
        const fastestPerMB = summaries.reduce((best, s) =>
            s.timePer1MB_ms < best.timePer1MB_ms && s.timePer1MB_ms > 0 ? s : best
        );
        console.log(`  Fastest per MB:           ${fastestPerMB.sessionId} (${formatDuration(fastestPerMB.timePer1MB_ms)})`);

        // Calculate speedup ratios
        const worstOcr = summaries.reduce((worst, s) =>
            s.ocrAvgCharsPerSec < worst.ocrAvgCharsPerSec && s.ocrAvgCharsPerSec > 0 ? s : worst
        );
        const worstEmbed = summaries.reduce((worst, s) =>
            s.embedAvgCharsPerSec < worst.embedAvgCharsPerSec && s.embedAvgCharsPerSec > 0 ? s : worst
        );

        if (bestOcr !== worstOcr && worstOcr.ocrAvgCharsPerSec > 0) {
            const ocrSpeedup = (bestOcr.ocrAvgCharsPerSec / worstOcr.ocrAvgCharsPerSec).toFixed(2);
            console.log(`  OCR Speedup (best/worst): ${ocrSpeedup}x`);
        }

        if (bestEmbed !== worstEmbed && worstEmbed.embedAvgCharsPerSec > 0) {
            const embedSpeedup = (bestEmbed.embedAvgCharsPerSec / worstEmbed.embedAvgCharsPerSec).toFixed(2);
            console.log(`  Embed Speedup (best/worst): ${embedSpeedup}x`);
        }
    }

    // System Analysis (group by system)
    if (uniqueSystems.size > 1) {
        console.log(chalk.cyan('\n[System Analysis]'));

        const systemGroups = new Map<string, SessionSummary[]>();
        for (const s of summaries) {
            const key = `${s.platform}|${s.cpuModel}`;
            if (!systemGroups.has(key)) {
                systemGroups.set(key, []);
            }
            systemGroups.get(key)!.push(s);
        }

        for (const [key, group] of systemGroups) {
            const [platform, cpu] = key.split('|');
            const avgOcr = group.reduce((sum, s) => sum + s.ocrAvgCharsPerSec, 0) / group.length;
            const avgEmbed = group.reduce((sum, s) => sum + s.embedAvgCharsPerSec, 0) / group.length;
            const avgTimePerMB = group.reduce((sum, s) => sum + s.timePer1MB_ms, 0) / group.length;

            console.log(chalk.yellow(`\n  ${platform} - ${cpu.substring(0, 40)}`));
            console.log(`    Sessions:           ${group.length}`);
            console.log(`    GPU Available:      ${group[0].gpuAvailable ? 'Yes' : 'No'}`);
            console.log(`    Avg OCR Speed:      ${formatNumber(Math.round(avgOcr))} chars/s`);
            console.log(`    Avg Embed Speed:    ${formatNumber(Math.round(avgEmbed))} chars/s`);
            console.log(`    Avg Time per MB:    ${formatDuration(avgTimePerMB)}`);
        }
    }

    // Recommendations
    console.log(chalk.cyan('\n[Recommendations]'));
    const recommendations: string[] = [];

    // Check for GPU usage
    const hasGpu = summaries.some(s => s.gpuAvailable);
    const gpuUsed = summaries.some(s => s.gpuAvailable && s.gpuInfo?.includes('NVIDIA'));
    if (hasGpu && !gpuUsed) {
        recommendations.push('GPU detected but NVIDIA CUDA not used. Consider using inception-gpu for faster embeddings.');
    }

    // Check for slow embedding
    const avgEmbedSpeed = summaries.reduce((sum, s) => sum + s.embedAvgCharsPerSec, 0) / summaries.length;
    if (avgEmbedSpeed < 1000) {
        recommendations.push('Embedding speed is low. Consider using GPU acceleration or increasing batch size.');
    }

    // Check for variance
    if (summaries.length > 1) {
        const speeds = summaries.map(s => s.embedAvgCharsPerSec);
        const maxSpeed = Math.max(...speeds);
        const minSpeed = Math.min(...speeds);
        if (maxSpeed > minSpeed * 2) {
            recommendations.push('High variance in embedding speeds detected. System configurations may differ significantly.');
        }
    }

    if (recommendations.length === 0) {
        recommendations.push('Performance looks good. No specific recommendations.');
    }

    for (const rec of recommendations) {
        console.log(`  - ${rec}`);
    }

    // Export summary
    console.log(chalk.cyan('\n[Export]'));
    const exportPath = path.join('logs', 'benchmark-summary.json');
    const comparison: BenchmarkComparison = {
        sessions,
        comparison: {
            fastestOcr: {
                sessionId: summaries.reduce((best, s) =>
                    s.ocrAvgCharsPerSec > best.ocrAvgCharsPerSec ? s : best
                ).sessionId,
                avgTimeMs: summaries.reduce((best, s) =>
                    s.ocrAvgCharsPerSec > best.ocrAvgCharsPerSec ? s : best
                ).ocrTimeMs / summaries.reduce((best, s) =>
                    s.ocrAvgCharsPerSec > best.ocrAvgCharsPerSec ? s : best
                ).filesProcessed
            },
            fastestEmbed: {
                sessionId: summaries.reduce((best, s) =>
                    s.embedAvgCharsPerSec > best.embedAvgCharsPerSec ? s : best
                ).sessionId,
                avgTimeMs: summaries.reduce((best, s) =>
                    s.embedAvgCharsPerSec > best.embedAvgCharsPerSec ? s : best
                ).embedTimeMs / summaries.reduce((best, s) =>
                    s.embedAvgCharsPerSec > best.embedAvgCharsPerSec ? s : best
                ).filesProcessed
            },
            fastestOverall: {
                sessionId: summaries.reduce((best, s) =>
                    s.timePer1MB_ms < best.timePer1MB_ms && s.timePer1MB_ms > 0 ? s : best
                ).sessionId,
                totalTimeMs: summaries.reduce((best, s) =>
                    s.timePer1MB_ms < best.timePer1MB_ms && s.timePer1MB_ms > 0 ? s : best
                ).totalTimeMs
            },
            recommendations
        }
    };

    fs.writeFileSync(exportPath, JSON.stringify(comparison, null, 2));
    console.log(`  Summary exported to: ${exportPath}`);

    console.log(chalk.blue.bold('\n========================================'));
    console.log(chalk.blue.bold('  Analysis Complete'));
    console.log(chalk.blue.bold('========================================'));
}

export function analyzeBenchmarks(folder: string): void {
    console.log(chalk.blue(`\n[Benchmark Analyzer]`));
    console.log(chalk.gray(`  Loading sessions from: ${folder}`));

    const sessions = loadSessions(folder);

    if (sessions.length === 0) {
        console.log(chalk.yellow('\n[INFO] No benchmark sessions found.'));
        console.log(chalk.gray('  Run the demo to generate benchmark data:'));
        console.log(chalk.gray('    docker compose run --rm client demo'));
        return;
    }

    console.log(chalk.green(`  Found ${sessions.length} session(s)`));
    printSessionComparison(sessions);
}

// Allow running directly
if (import.meta.main) {
    const folder = process.argv[2] || 'logs';
    analyzeBenchmarks(folder);
}
