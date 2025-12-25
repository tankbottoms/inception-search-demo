# Benchmark Session Format Specification

This document describes the JSON format for benchmark session files. Use this specification to instrument applications for consistent performance comparison across systems.

## Purpose

Benchmark sessions capture:

- System hardware specifications for comparison across machines
- Per-file processing metrics for identifying bottlenecks
- Aggregated statistics for high-level performance analysis
- Time projections for capacity planning

## File Naming

Session files should be saved with timestamp-based names:

```
YYYYMMDD-HHMMSS.json
```

Example: `20251224-143022.json`

## JSON Schema

### Root Object: `BenchmarkSession`

```json
{
  "sessionId": "20251224-143022",
  "timestamp": "20251224-143022",
  "timestampIso": "2025-12-24T14:30:22.000Z",
  "system": { ... },
  "config": { ... },
  "files": [ ... ],
  "stats": { ... },
  "search": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| sessionId | string | Unique session identifier (timestamp format) |
| timestamp | string | Same as sessionId for compatibility |
| timestampIso | string | ISO 8601 timestamp |
| system | SystemInfo | Hardware and platform details |
| config | object | Configuration used for this run |
| files | FileMetrics[] | Per-file processing metrics |
| stats | SessionStats | Aggregated statistics |
| search | SearchMetrics | Optional search performance data |

### SystemInfo

```json
{
  "platform": "darwin",
  "arch": "arm64",
  "cpuModel": "Apple M1 Pro",
  "cpuCores": 10,
  "totalMemoryGB": 32.0,
  "freeMemoryGB": 16.5,
  "nodeVersion": "v20.10.0",
  "bunVersion": "1.0.0",
  "hostname": "machine-name",
  "osRelease": "23.0.0",
  "gpuAvailable": true,
  "gpuInfo": "Apple M1 Pro"
}
```

| Field | Type | Description |
|-------|------|-------------|
| platform | string | OS platform: darwin, linux, win32 |
| arch | string | CPU architecture: arm64, x64 |
| cpuModel | string | CPU model name |
| cpuCores | number | Number of CPU cores |
| totalMemoryGB | number | Total system RAM in GB |
| freeMemoryGB | number | Available RAM at session start |
| nodeVersion | string | Node.js version |
| bunVersion | string | Bun version (optional) |
| hostname | string | Machine hostname |
| osRelease | string | OS kernel/release version |
| gpuAvailable | boolean | Whether GPU is detected |
| gpuInfo | string | GPU model name (optional) |

### FileMetrics

Use file hashes instead of filenames for privacy when sharing benchmarks.

```json
{
  "fileHash": "abc123def456...",
  "fileSizeBytes": 5242880,
  "fileSizeMB": 5.0,
  "pageCount": 25,
  "estimatedChars": 50000,
  "ocrStartTime": 1703423422000,
  "ocrEndTime": 1703423430000,
  "ocrDurationMs": 8000,
  "ocrOutputChars": 48500,
  "ocrOutputSizeBytes": 52000,
  "rawTextChars": 45000,
  "textSimilarityPercent": 95.5,
  "textDifferencePercent": 4.5,
  "embedStartTime": 1703423430000,
  "embedEndTime": 1703423442000,
  "embedDurationMs": 12000,
  "chunkCount": 15,
  "tokensEstimate": 12125,
  "ocrCharsPerSecond": 6062,
  "embedCharsPerSecond": 4041,
  "embedTokensPerSecond": 1010
}
```

| Field | Type | Description |
|-------|------|-------------|
| fileHash | string | SHA256 hash (first 16 chars) for privacy |
| fileSizeBytes | number | Original file size in bytes |
| fileSizeMB | number | Original file size in megabytes |
| pageCount | number | Number of pages (estimated or actual) |
| estimatedChars | number | Pre-OCR character estimate |
| ocrStartTime | number | OCR start timestamp (ms since epoch) |
| ocrEndTime | number | OCR end timestamp |
| ocrDurationMs | number | OCR processing time in milliseconds |
| ocrOutputChars | number | Characters extracted by OCR |
| ocrOutputSizeBytes | number | OCR output size in bytes (UTF-8) |
| rawTextChars | number | Characters from raw text extraction (optional) |
| textSimilarityPercent | number | Similarity between OCR and raw text (optional) |
| textDifferencePercent | number | Difference percentage (optional) |
| embedStartTime | number | Embedding start timestamp |
| embedEndTime | number | Embedding end timestamp |
| embedDurationMs | number | Embedding time in milliseconds |
| chunkCount | number | Number of text chunks created |
| tokensEstimate | number | Estimated token count (chars / 4) |
| ocrCharsPerSecond | number | OCR throughput |
| embedCharsPerSecond | number | Embedding throughput (chars) |
| embedTokensPerSecond | number | Embedding throughput (tokens) |

### SessionStats

Aggregated statistics across all files.

```json
{
  "totalFiles": 5,
  "totalSizeMB": 25.5,
  "totalPages": 125,
  "totalChars": 250000,
  "totalChunks": 75,
  "totalDurationMs": 120000,
  "totalOcrTimeMs": 40000,
  "totalEmbedTimeMs": 80000,
  "ocrAvgTimeMs": 8000,
  "ocrMinTimeMs": 5000,
  "ocrMaxTimeMs": 12000,
  "ocrAvgCharsPerSecond": 6250,
  "embedAvgTimeMs": 16000,
  "embedMinTimeMs": 10000,
  "embedMaxTimeMs": 25000,
  "embedAvgCharsPerSecond": 3125,
  "embedAvgTokensPerSecond": 781,
  "estimatedTimePer1MB_ms": 4706,
  "estimatedTimePer100MB_ms": 470600,
  "estimatedTimePer1GB_ms": 4818944,
  "estimatedEmbedPer100Chars_ms": 32,
  "estimatedEmbedPer1000Chars_ms": 320,
  "estimatedEmbedPer100Tokens_ms": 128
}
```

| Field | Type | Description |
|-------|------|-------------|
| totalFiles | number | Number of files processed |
| totalSizeMB | number | Sum of all file sizes |
| totalPages | number | Sum of all page counts |
| totalChars | number | Sum of all extracted characters |
| totalChunks | number | Sum of all chunks created |
| totalDurationMs | number | Total processing time |
| totalOcrTimeMs | number | Sum of all OCR times |
| totalEmbedTimeMs | number | Sum of all embedding times |
| ocrAvgTimeMs | number | Average OCR time per file |
| ocrMinTimeMs | number | Fastest OCR time |
| ocrMaxTimeMs | number | Slowest OCR time |
| ocrAvgCharsPerSecond | number | Average OCR throughput |
| embedAvgTimeMs | number | Average embedding time per file |
| embedMinTimeMs | number | Fastest embedding time |
| embedMaxTimeMs | number | Slowest embedding time |
| embedAvgCharsPerSecond | number | Average embedding throughput (chars) |
| embedAvgTokensPerSecond | number | Average embedding throughput (tokens) |
| estimatedTimePer1MB_ms | number | Projected time for 1 MB |
| estimatedTimePer100MB_ms | number | Projected time for 100 MB |
| estimatedTimePer1GB_ms | number | Projected time for 1 GB |
| estimatedEmbedPer100Chars_ms | number | Embedding time per 100 characters |
| estimatedEmbedPer1000Chars_ms | number | Embedding time per 1000 characters |
| estimatedEmbedPer100Tokens_ms | number | Embedding time per 100 tokens |

### SearchMetrics (Optional)

If a search was performed during the session.

```json
{
  "query": "fraud allegations",
  "queryEmbedTimeMs": 150,
  "similaritySearchTimeMs": 25,
  "totalSearchTimeMs": 175,
  "chunksSearched": 75,
  "resultsReturned": 5,
  "throughputChunksPerSec": 3000
}
```

| Field | Type | Description |
|-------|------|-------------|
| query | string | Search query text |
| queryEmbedTimeMs | number | Time to embed the query |
| similaritySearchTimeMs | number | Time for vector similarity search |
| totalSearchTimeMs | number | Total search time including overhead |
| chunksSearched | number | Number of chunks compared |
| resultsReturned | number | Number of results returned |
| throughputChunksPerSec | number | Search throughput |

## Instrumentation Guidelines

### Timing

- Use `performance.now()` for high-resolution timing
- Record start time before operation, end time after
- Calculate duration as `endTime - startTime`

### File Hashing

For privacy, use SHA256 hash instead of filenames:

```typescript
import crypto from 'crypto';

function computeFileHash(filePath: string): string {
    const buffer = fs.readFileSync(filePath);
    return crypto.createHash('sha256')
        .update(buffer)
        .digest('hex')
        .substring(0, 16);
}
```

### Token Estimation

Estimate tokens as characters divided by 4:

```typescript
const tokensEstimate = Math.round(charCount / 4);
```

### Throughput Calculation

```typescript
const charsPerSecond = durationMs > 0
    ? (charCount / (durationMs / 1000))
    : 0;
```

### Projections

Calculate projections from actual measurements:

```typescript
const avgTimePerChar = totalChars > 0
    ? totalTimeMs / totalChars
    : 0;

const estimatedTimePer1MB = avgTimePerMB;
const estimatedTimePer100MB = avgTimePerMB * 100;
const estimatedTimePer1GB = avgTimePerMB * 1024;
```

## Example Complete Session

```json
{
  "sessionId": "20251224-143022",
  "timestamp": "20251224-143022",
  "timestampIso": "2025-12-24T14:30:22.000Z",
  "system": {
    "platform": "darwin",
    "arch": "arm64",
    "cpuModel": "Apple M1 Pro",
    "cpuCores": 10,
    "totalMemoryGB": 32,
    "freeMemoryGB": 18.5,
    "nodeVersion": "v20.10.0",
    "bunVersion": "1.0.0",
    "hostname": "dev-machine",
    "osRelease": "23.0.0",
    "gpuAvailable": true,
    "gpuInfo": "Apple M1 Pro"
  },
  "config": {
    "inceptionUrl": "http://localhost:8005",
    "modelName": "freelawproject/modernbert-embed-base_finetune_512",
    "pdfCount": 5,
    "searchQuery": "fraud"
  },
  "files": [
    {
      "fileHash": "abc123def456",
      "fileSizeBytes": 5242880,
      "fileSizeMB": 5.0,
      "pageCount": 25,
      "estimatedChars": 50000,
      "ocrStartTime": 1703423422000,
      "ocrEndTime": 1703423430000,
      "ocrDurationMs": 8000,
      "ocrOutputChars": 48500,
      "ocrOutputSizeBytes": 52000,
      "embedStartTime": 1703423430000,
      "embedEndTime": 1703423442000,
      "embedDurationMs": 12000,
      "chunkCount": 15,
      "tokensEstimate": 12125,
      "ocrCharsPerSecond": 6062,
      "embedCharsPerSecond": 4041,
      "embedTokensPerSecond": 1010
    }
  ],
  "stats": {
    "totalFiles": 1,
    "totalSizeMB": 5.0,
    "totalPages": 25,
    "totalChars": 48500,
    "totalChunks": 15,
    "totalDurationMs": 20000,
    "totalOcrTimeMs": 8000,
    "totalEmbedTimeMs": 12000,
    "ocrAvgTimeMs": 8000,
    "ocrMinTimeMs": 8000,
    "ocrMaxTimeMs": 8000,
    "ocrAvgCharsPerSecond": 6062,
    "embedAvgTimeMs": 12000,
    "embedMinTimeMs": 12000,
    "embedMaxTimeMs": 12000,
    "embedAvgCharsPerSecond": 4041,
    "embedAvgTokensPerSecond": 1010,
    "estimatedTimePer1MB_ms": 4000,
    "estimatedTimePer100MB_ms": 400000,
    "estimatedTimePer1GB_ms": 4096000,
    "estimatedEmbedPer100Chars_ms": 25,
    "estimatedEmbedPer1000Chars_ms": 247,
    "estimatedEmbedPer100Tokens_ms": 119
  },
  "search": {
    "query": "fraud",
    "queryEmbedTimeMs": 120,
    "similaritySearchTimeMs": 15,
    "totalSearchTimeMs": 135,
    "chunksSearched": 15,
    "resultsReturned": 5,
    "throughputChunksPerSec": 1000
  }
}
```

## Comparison Analysis

When comparing sessions across systems:

1. **Group by system** - Compare sessions from the same hardware
2. **Normalize by size** - Use per-MB or per-char metrics for fair comparison
3. **Consider variance** - Multiple runs per system help identify noise
4. **Check GPU utilization** - GPU sessions should be significantly faster

### Key Comparison Metrics

| Metric | Purpose |
|--------|---------|
| embedAvgCharsPerSecond | Primary embedding performance |
| embedAvgTokensPerSecond | Token-based comparison |
| estimatedTimePer1MB_ms | Capacity planning |
| ocrAvgCharsPerSecond | OCR comparison (if applicable) |

### Speedup Calculation

```typescript
const speedup = fasterSession.embedAvgCharsPerSecond /
                slowerSession.embedAvgCharsPerSecond;
```

## Example Session JSON File

Below is a complete example of a benchmark session JSON file. Use this as a reference when implementing instrumentation in other applications.

```json
{
  "sessionId": "20251224-170213",
  "timestamp": "20251224-170213",
  "timestampIso": "2025-12-24T17:02:13.456Z",
  "system": {
    "platform": "darwin",
    "arch": "arm64",
    "cpuModel": "Apple M1 Pro",
    "cpuCores": 10,
    "totalMemoryGB": 32,
    "freeMemoryGB": 18.5,
    "nodeVersion": "v20.10.0",
    "bunVersion": "1.1.42",
    "hostname": "dev-macbook",
    "osRelease": "23.6.0",
    "gpuAvailable": true,
    "gpuInfo": "Apple M1 Pro"
  },
  "config": {
    "inceptionUrl": "http://localhost:8005",
    "modelName": "freelawproject/modernbert-embed-base_finetune_512",
    "pdfCount": 3,
    "searchQuery": "fraud"
  },
  "files": [
    {
      "fileHash": "89b0e1188154be8d",
      "fileSizeBytes": 2456789,
      "fileSizeMB": 2.34,
      "pageCount": 15,
      "estimatedChars": 30000,
      "ocrStartTime": 1703437333456,
      "ocrEndTime": 1703437341890,
      "ocrDurationMs": 8434,
      "ocrOutputChars": 28750,
      "ocrOutputSizeBytes": 31200,
      "rawTextChars": 27500,
      "textSimilarityPercent": 95.6,
      "textDifferencePercent": 4.4,
      "embedStartTime": 1703437341891,
      "embedEndTime": 1703437354123,
      "embedDurationMs": 12232,
      "chunkCount": 12,
      "tokensEstimate": 7188,
      "ocrCharsPerSecond": 3409,
      "embedCharsPerSecond": 2350,
      "embedTokensPerSecond": 588
    },
    {
      "fileHash": "55f250b7e6600b9f",
      "fileSizeBytes": 5123456,
      "fileSizeMB": 4.89,
      "pageCount": 32,
      "estimatedChars": 64000,
      "ocrStartTime": 1703437354124,
      "ocrEndTime": 1703437372456,
      "ocrDurationMs": 18332,
      "ocrOutputChars": 61200,
      "ocrOutputSizeBytes": 66100,
      "rawTextChars": 59800,
      "textSimilarityPercent": 97.7,
      "textDifferencePercent": 2.3,
      "embedStartTime": 1703437372457,
      "embedEndTime": 1703437398789,
      "embedDurationMs": 26332,
      "chunkCount": 25,
      "tokensEstimate": 15300,
      "ocrCharsPerSecond": 3339,
      "embedCharsPerSecond": 2324,
      "embedTokensPerSecond": 581
    },
    {
      "fileHash": "c33a1d0b0efbda88",
      "fileSizeBytes": 1234567,
      "fileSizeMB": 1.18,
      "pageCount": 8,
      "estimatedChars": 16000,
      "ocrStartTime": 1703437398790,
      "ocrEndTime": 1703437403456,
      "ocrDurationMs": 4666,
      "ocrOutputChars": 15200,
      "ocrOutputSizeBytes": 16400,
      "rawTextChars": 14900,
      "textSimilarityPercent": 98.0,
      "textDifferencePercent": 2.0,
      "embedStartTime": 1703437403457,
      "embedEndTime": 1703437410123,
      "embedDurationMs": 6666,
      "chunkCount": 6,
      "tokensEstimate": 3800,
      "ocrCharsPerSecond": 3259,
      "embedCharsPerSecond": 2280,
      "embedTokensPerSecond": 570
    }
  ],
  "stats": {
    "totalFiles": 3,
    "totalSizeMB": 8.41,
    "totalPages": 55,
    "totalChars": 105150,
    "totalChunks": 43,
    "totalDurationMs": 76667,
    "totalOcrTimeMs": 31432,
    "totalEmbedTimeMs": 45230,
    "ocrAvgTimeMs": 10477,
    "ocrMinTimeMs": 4666,
    "ocrMaxTimeMs": 18332,
    "ocrAvgCharsPerSecond": 3346,
    "embedAvgTimeMs": 15077,
    "embedMinTimeMs": 6666,
    "embedMaxTimeMs": 26332,
    "embedAvgCharsPerSecond": 2325,
    "embedAvgTokensPerSecond": 581,
    "estimatedTimePer1MB_ms": 9116,
    "estimatedTimePer100MB_ms": 911600,
    "estimatedTimePer1GB_ms": 9334784,
    "estimatedEmbedPer100Chars_ms": 43,
    "estimatedEmbedPer1000Chars_ms": 430,
    "estimatedEmbedPer100Tokens_ms": 172
  },
  "search": {
    "query": "fraud",
    "queryEmbedTimeMs": 145,
    "similaritySearchTimeMs": 18,
    "totalSearchTimeMs": 163,
    "chunksSearched": 43,
    "resultsReturned": 5,
    "throughputChunksPerSec": 2389
  }
}
```

This example shows a session with 3 PDF files processed on an Apple M1 Pro. Note that:

- File names are replaced with SHA256 hashes (first 16 characters) for privacy
- All timestamps use milliseconds since epoch for precision
- Throughput metrics are calculated from actual measurements
- The search section is optional and only included when a search was performed
