// Benchmark and Session Types for Inception Demo

export interface SystemInfo {
    platform: string;
    arch: string;
    cpuModel: string;
    cpuCores: number;
    totalMemoryGB: number;
    freeMemoryGB: number;
    nodeVersion: string;
    bunVersion?: string;
    hostname: string;
    osRelease: string;
    gpuAvailable: boolean;
    gpuInfo?: string;
}

export interface FileMetrics {
    fileHash: string;  // SHA256 hash for privacy
    fileSizeBytes: number;
    fileSizeMB: number;
    pageCount: number;
    estimatedChars: number;

    // OCR Metrics
    ocrStartTime: number;
    ocrEndTime: number;
    ocrDurationMs: number;
    ocrOutputChars: number;
    ocrOutputSizeBytes: number;

    // Text extraction comparison (if available)
    rawTextChars?: number;
    textSimilarityPercent?: number;
    textDifferencePercent?: number;

    // Embedding Metrics
    embedStartTime: number;
    embedEndTime: number;
    embedDurationMs: number;
    chunkCount: number;
    tokensEstimate: number;  // rough estimate based on chars/4

    // Computed speeds
    ocrCharsPerSecond: number;
    embedCharsPerSecond: number;
    embedTokensPerSecond: number;
}

export interface SessionStats {
    totalFiles: number;
    totalSizeMB: number;
    totalPages: number;
    totalChars: number;
    totalChunks: number;

    // Timing
    totalDurationMs: number;
    totalOcrTimeMs: number;
    totalEmbedTimeMs: number;

    // OCR Stats
    ocrAvgTimeMs: number;
    ocrMinTimeMs: number;
    ocrMaxTimeMs: number;
    ocrAvgCharsPerSecond: number;

    // Embedding Stats
    embedAvgTimeMs: number;
    embedMinTimeMs: number;
    embedMaxTimeMs: number;
    embedAvgCharsPerSecond: number;
    embedAvgTokensPerSecond: number;

    // Projections
    estimatedTimePer1MB_ms: number;
    estimatedTimePer100MB_ms: number;
    estimatedTimePer1GB_ms: number;
    estimatedEmbedPer100Chars_ms: number;
    estimatedEmbedPer1000Chars_ms: number;
    estimatedEmbedPer100Tokens_ms: number;
}

export interface SearchMetrics {
    query: string;
    queryEmbedTimeMs: number;
    similaritySearchTimeMs: number;
    totalSearchTimeMs: number;
    chunksSearched: number;
    resultsReturned: number;
    throughputChunksPerSec: number;
}

export interface BenchmarkSession {
    sessionId: string;
    timestamp: string;
    timestampIso: string;

    // System Info
    system: SystemInfo;

    // Configuration
    config: {
        inceptionUrl: string;
        modelName: string;
        pdfCount: number;
        searchQuery?: string;
    };

    // File Metrics
    files: FileMetrics[];

    // Aggregated Stats
    stats: SessionStats;

    // Search Metrics (if search was performed)
    search?: SearchMetrics;
}

export interface BenchmarkComparison {
    sessions: BenchmarkSession[];
    comparison: {
        fastestOcr: { sessionId: string; avgTimeMs: number };
        fastestEmbed: { sessionId: string; avgTimeMs: number };
        fastestOverall: { sessionId: string; totalTimeMs: number };
        recommendations: string[];
    };
}
