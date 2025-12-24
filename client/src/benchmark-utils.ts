// Benchmark Utilities for Inception Demo

import os from 'os';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';
import type { SystemInfo, FileMetrics, SessionStats, BenchmarkSession } from './types.js';

const LOGS_DIR = 'logs';

export function getSystemInfo(): SystemInfo {
    const cpus = os.cpus();

    let cpuModel = cpus[0]?.model || '';
    let gpuAvailable = false;
    let gpuInfo: string | undefined;

    // Detect if running in Docker
    const isDocker = fs.existsSync('/.dockerenv') ||
        (fs.existsSync('/proc/1/cgroup') &&
         fs.readFileSync('/proc/1/cgroup', 'utf-8').includes('docker'));

    // Try to detect CPU and GPU
    try {
        if (process.platform === 'darwin') {
            // macOS native - get Apple Silicon chip name via sysctl
            if (!cpuModel || cpuModel === 'Unknown' || cpuModel.trim() === '') {
                try {
                    const chipName = execSync('sysctl -n machdep.cpu.brand_string 2>/dev/null', { encoding: 'utf-8' }).trim();
                    if (chipName) {
                        cpuModel = chipName;
                    }
                } catch {
                    // Try alternative method for Apple Silicon
                    try {
                        const hwModel = execSync('sysctl -n hw.model 2>/dev/null', { encoding: 'utf-8' }).trim();
                        if (hwModel) {
                            cpuModel = hwModel;
                        }
                    } catch {
                        // Fallback to system_profiler
                        try {
                            const spOutput = execSync('system_profiler SPHardwareDataType 2>/dev/null', { encoding: 'utf-8' });
                            const chipMatch = spOutput.match(/Chip:\s*(.+)/);
                            if (chipMatch) {
                                cpuModel = chipMatch[1].trim();
                            } else {
                                const procMatch = spOutput.match(/Processor Name:\s*(.+)/);
                                if (procMatch) {
                                    cpuModel = procMatch[1].trim();
                                }
                            }
                        } catch {
                            // Give up on CPU detection
                        }
                    }
                }
            }

            // macOS - check for Metal GPU
            const gpuOutput = execSync('system_profiler SPDisplaysDataType 2>/dev/null', { encoding: 'utf-8' });
            const gpuMatch = gpuOutput.match(/Chipset Model:\s*(.+)/);
            if (gpuMatch) {
                gpuInfo = gpuMatch[1].trim();
                gpuAvailable = gpuInfo.toLowerCase().includes('apple') || gpuInfo.toLowerCase().includes('nvidia');
            }
        } else if (process.platform === 'linux') {
            // Linux or Docker container
            if (!cpuModel || cpuModel === 'Unknown' || cpuModel.trim() === '') {
                // Try lscpu first
                try {
                    const lscpuOutput = execSync('lscpu 2>/dev/null', { encoding: 'utf-8' });
                    const modelMatch = lscpuOutput.match(/Model name:\s*(.+)/);
                    if (modelMatch) {
                        cpuModel = modelMatch[1].trim();
                    }
                } catch {
                    // Keep trying
                }
            }

            // If still empty and on ARM64, likely Docker on Apple Silicon
            if ((!cpuModel || cpuModel.trim() === '') && process.arch === 'arm64') {
                // Check /proc/cpuinfo for more details
                try {
                    const cpuinfo = fs.readFileSync('/proc/cpuinfo', 'utf-8');
                    const implMatch = cpuinfo.match(/CPU implementer\s*:\s*0x61/); // 0x61 = Apple
                    if (implMatch) {
                        // Running on Apple Silicon via Docker/QEMU
                        cpuModel = 'Apple Silicon (via Docker)';
                        gpuInfo = 'Apple Silicon GPU (host)';
                        gpuAvailable = true;
                    } else {
                        // Generic ARM64
                        const partMatch = cpuinfo.match(/CPU part\s*:\s*(.+)/);
                        if (partMatch) {
                            cpuModel = `ARM64 (${partMatch[1].trim()})`;
                        }
                    }
                } catch {
                    // Fallback
                }
            }

            // Check for NVIDIA GPU
            try {
                const nvidiaSmi = execSync('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null', { encoding: 'utf-8' });
                if (nvidiaSmi.trim()) {
                    gpuInfo = nvidiaSmi.trim();
                    gpuAvailable = true;
                }
            } catch {
                // No NVIDIA GPU
            }
        }
    } catch {
        // Ignore detection errors
    }

    // Final fallback with architecture info
    if (!cpuModel || cpuModel.trim() === '') {
        if (process.arch === 'arm64' && isDocker) {
            cpuModel = 'ARM64 CPU (Docker)';
        } else {
            cpuModel = `${process.arch} CPU`;
        }
    }

    // Add Docker indicator to hostname if in container
    let hostname = os.hostname();
    const platformDisplay = isDocker ? 'linux (Docker)' : process.platform;

    let bunVersion: string | undefined;
    try {
        bunVersion = execSync('bun --version 2>/dev/null', { encoding: 'utf-8' }).trim();
    } catch {
        // Bun not available
    }

    return {
        platform: platformDisplay,
        arch: process.arch,
        cpuModel,
        cpuCores: cpus.length,
        totalMemoryGB: Math.round(os.totalmem() / (1024 ** 3) * 100) / 100,
        freeMemoryGB: Math.round(os.freemem() / (1024 ** 3) * 100) / 100,
        nodeVersion: process.version,
        bunVersion,
        hostname,
        osRelease: os.release(),
        gpuAvailable,
        gpuInfo
    };
}

export function computeFileHash(filePath: string): string {
    const fileBuffer = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(fileBuffer).digest('hex').substring(0, 16);
}

export function generateSessionId(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hour = String(now.getHours()).padStart(2, '0');
    const minute = String(now.getMinutes()).padStart(2, '0');
    const second = String(now.getSeconds()).padStart(2, '0');
    return `${year}${month}${day}-${hour}${minute}${second}`;
}

export function calculateSessionStats(files: FileMetrics[]): SessionStats {
    if (files.length === 0) {
        return {
            totalFiles: 0,
            totalSizeMB: 0,
            totalPages: 0,
            totalChars: 0,
            totalChunks: 0,
            totalDurationMs: 0,
            totalOcrTimeMs: 0,
            totalEmbedTimeMs: 0,
            ocrAvgTimeMs: 0,
            ocrMinTimeMs: 0,
            ocrMaxTimeMs: 0,
            ocrAvgCharsPerSecond: 0,
            embedAvgTimeMs: 0,
            embedMinTimeMs: 0,
            embedMaxTimeMs: 0,
            embedAvgCharsPerSecond: 0,
            embedAvgTokensPerSecond: 0,
            estimatedTimePer1MB_ms: 0,
            estimatedTimePer100MB_ms: 0,
            estimatedTimePer1GB_ms: 0,
            estimatedEmbedPer100Chars_ms: 0,
            estimatedEmbedPer1000Chars_ms: 0,
            estimatedEmbedPer100Tokens_ms: 0
        };
    }

    const totalSizeMB = files.reduce((sum, f) => sum + f.fileSizeMB, 0);
    const totalPages = files.reduce((sum, f) => sum + f.pageCount, 0);
    const totalChars = files.reduce((sum, f) => sum + f.ocrOutputChars, 0);
    const totalChunks = files.reduce((sum, f) => sum + f.chunkCount, 0);
    const totalOcrTimeMs = files.reduce((sum, f) => sum + f.ocrDurationMs, 0);
    const totalEmbedTimeMs = files.reduce((sum, f) => sum + f.embedDurationMs, 0);
    const totalDurationMs = totalOcrTimeMs + totalEmbedTimeMs;

    const ocrTimes = files.map(f => f.ocrDurationMs);
    const embedTimes = files.map(f => f.embedDurationMs);

    const ocrCharsPerSec = files.map(f => f.ocrCharsPerSecond).filter(v => v > 0);
    const embedCharsPerSec = files.map(f => f.embedCharsPerSecond).filter(v => v > 0);
    const embedTokensPerSec = files.map(f => f.embedTokensPerSecond).filter(v => v > 0);

    const avgOcrCharsPerSecond = ocrCharsPerSec.length > 0
        ? ocrCharsPerSec.reduce((a, b) => a + b, 0) / ocrCharsPerSec.length
        : 0;
    const avgEmbedCharsPerSecond = embedCharsPerSec.length > 0
        ? embedCharsPerSec.reduce((a, b) => a + b, 0) / embedCharsPerSec.length
        : 0;
    const avgEmbedTokensPerSecond = embedTokensPerSec.length > 0
        ? embedTokensPerSec.reduce((a, b) => a + b, 0) / embedTokensPerSec.length
        : 0;

    // Estimate time per MB based on actual data
    const avgTimePerMB = totalSizeMB > 0 ? totalDurationMs / totalSizeMB : 0;

    // Estimate embedding time per characters/tokens
    const avgEmbedTimePerChar = totalChars > 0 ? totalEmbedTimeMs / totalChars : 0;
    const totalTokens = files.reduce((sum, f) => sum + f.tokensEstimate, 0);
    const avgEmbedTimePerToken = totalTokens > 0 ? totalEmbedTimeMs / totalTokens : 0;

    return {
        totalFiles: files.length,
        totalSizeMB: Math.round(totalSizeMB * 100) / 100,
        totalPages,
        totalChars,
        totalChunks,
        totalDurationMs: Math.round(totalDurationMs),
        totalOcrTimeMs: Math.round(totalOcrTimeMs),
        totalEmbedTimeMs: Math.round(totalEmbedTimeMs),
        ocrAvgTimeMs: Math.round(totalOcrTimeMs / files.length),
        ocrMinTimeMs: Math.round(Math.min(...ocrTimes)),
        ocrMaxTimeMs: Math.round(Math.max(...ocrTimes)),
        ocrAvgCharsPerSecond: Math.round(avgOcrCharsPerSecond),
        embedAvgTimeMs: Math.round(totalEmbedTimeMs / files.length),
        embedMinTimeMs: Math.round(Math.min(...embedTimes)),
        embedMaxTimeMs: Math.round(Math.max(...embedTimes)),
        embedAvgCharsPerSecond: Math.round(avgEmbedCharsPerSecond),
        embedAvgTokensPerSecond: Math.round(avgEmbedTokensPerSecond),
        estimatedTimePer1MB_ms: Math.round(avgTimePerMB),
        estimatedTimePer100MB_ms: Math.round(avgTimePerMB * 100),
        estimatedTimePer1GB_ms: Math.round(avgTimePerMB * 1024),
        estimatedEmbedPer100Chars_ms: Math.round(avgEmbedTimePerChar * 100),
        estimatedEmbedPer1000Chars_ms: Math.round(avgEmbedTimePerChar * 1000),
        estimatedEmbedPer100Tokens_ms: Math.round(avgEmbedTimePerToken * 100)
    };
}

export function ensureLogsDir(): string {
    if (!fs.existsSync(LOGS_DIR)) {
        fs.mkdirSync(LOGS_DIR, { recursive: true });
    }
    return LOGS_DIR;
}

export function saveSession(session: BenchmarkSession): string {
    const logsDir = ensureLogsDir();
    const filename = `${session.sessionId}.json`;
    const filepath = path.join(logsDir, filename);
    fs.writeFileSync(filepath, JSON.stringify(session, null, 2));
    return filepath;
}

export function loadSession(filepath: string): BenchmarkSession {
    const content = fs.readFileSync(filepath, 'utf-8');
    return JSON.parse(content) as BenchmarkSession;
}

export function listSessions(): string[] {
    const logsDir = ensureLogsDir();
    return fs.readdirSync(logsDir)
        .filter(f => f.endsWith('.json'))
        .map(f => path.join(logsDir, f))
        .sort();
}

export function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(1);
    return `${minutes}m ${seconds}s`;
}

export function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatNumber(n: number): string {
    return n.toLocaleString();
}

// Estimate page count from file size (rough heuristic)
export function estimatePageCount(fileSizeBytes: number): number {
    // Average PDF page is roughly 100KB for scanned documents
    // This is a rough estimate - actual count should come from PDF parsing
    return Math.max(1, Math.round(fileSizeBytes / (100 * 1024)));
}

// Estimate token count from character count
export function estimateTokens(chars: number): number {
    // Rough estimate: ~4 chars per token for English text
    return Math.round(chars / 4);
}
