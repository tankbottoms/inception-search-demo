# Inception Demo - Document OCR and Semantic Search

A complete document processing and semantic search system combining **Mistral OCR**, **Free Law Project Inception embeddings**, and **vector similarity search**.

[Free Law Project Inception](https://github.com/freelawproject/inception/) provides a fine-tuned model based on their unprecedented database of legal documents. The transformer model is **freelawproject/modernbert-embed-base_finetune_512**. This repository demonstrates its capabilities on macOS M1 (CPU) and NVIDIA DGX Spark (GPU with CUDA).

From the [Free Law Project Inception](https://github.com/freelawproject/inception/) documentation:

> Inception is our microservice for generating embeddings from blocks of text.
>
> It is a high-performance FastAPI service that generates text embeddings using SentenceTransformers, specifically designed for processing legal documents and search queries. The service efficiently handles both short search queries and lengthy court opinions, generating semantic embeddings that can be used for document similarity matching and semantic search applications. It includes support for GPU acceleration when available.
>
> The service is optimized to handle two main use cases:
>
> - Embedding search queries: Quick, CPU-based processing for short search queries
> - Embedding court opinions: NVIDIA CUDA GPU-accelerated processing for longer legal documents, with intelligent text chunking to maintain context


## Quickstart

Get up and running in under a minute.

### Quick Start (3 Commands)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your MISTRAL_OCR_API_KEY

# 2. Build and start services
docker compose build client && docker compose up -d inception-cpu

# 3. Run full demo with all files
docker compose run --rm client demo --pdf-count 0
```

### Individual Commands

```bash
# Build client image (required after code changes)
docker compose build client

# Start embedding server (Apple Silicon / ARM64)
docker compose up -d inception-cpu

# Start embedding server (NVIDIA GPU)
docker compose --profile gpu up -d inception-gpu

# Run demo with 1 file (quick test)
docker compose run --rm client demo

# Run demo with all files
docker compose run --rm client demo --pdf-count 0

# Run demo with custom query
docker compose run --rm client demo --pdf-count 0 "securities fraud"

# Check server status
docker compose ps

# View server logs
docker compose logs -f inception-cpu

# Stop all services
docker compose down
```

### Automated Script

```bash
# Clone and run the demo
git clone https://github.com/your-repo/inception-demo.git
cd inception-demo
cp .env.example .env
# Edit .env and add your MISTRAL_OCR_API_KEY
./test-docker-stack.sh
```

The script automatically detects your platform (Apple Silicon M1/M2/M3 or NVIDIA GPU) and runs the appropriate backend.


## Overview

This demo showcases:

- **Mistral OCR API** - Converts PDFs and images to markdown ($2 per 1,000 pages, requires API key)
- **Inception Embedding Service** - Generates vector embeddings using ModernBERT (Docker service for CPU and GPU)
- **Semantic Search** - Finds relevant content using cosine similarity
- **Doctor Service** - Additional document processing capabilities from Free Law Project (untested)
- **CLI Tools** - Simple command-line client application for demos

## Features

- **Multi-format OCR**: PDFs, images, Office documents (via Mistral)
- **Semantic Search**: Context-aware document retrieval using a state-of-the-art fine-tuned model
- **PDF Text Comparison**: Extracts embedded PDF text and compares with OCR output (similarity %)
- **Docker-based**: Complete containerized stack
- **Platform Detection**: Automatic selection of CPU or GPU backend (Apple Silicon, NVIDIA)
- **OCR Export**: Saves OCR output as `.ocr.md` markdown files
- **Benchmark Statistics**: Detailed timing, performance metrics, and system detection
- **Privacy-Safe Benchmarks**: File names hashed with SHA256 for sharing


## Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- [Bun](https://bun.sh/) (optional, for local development)
- Mistral API Key (get one at [https://mistral.ai](https://mistral.ai))


## Usage

### Command Reference

The CLI provides four main commands.

#### 1. `demo` - Run Full Demo

Index files in the `files/` directory and perform a semantic search.

```bash
# Default: index 1 random file and search for "fraud"
docker compose run --rm client demo

# Index all files (use --pdf-count 0)
docker compose run --rm client demo --pdf-count 0

# Custom file count and query
docker compose run --rm client demo --pdf-count 5 "securities fraud"

# Add a specific file to the demo
docker compose run --rm client demo /path/to/document.pdf "custom query"
```

The demo extracts embedded PDF text before OCR and displays comparison statistics showing how similar the raw text is to the OCR output.

#### 2. `index` - Index Documents

Index a single file or directory of documents.

```bash
# Index a single PDF
docker compose run --rm client index "files/document.pdf"

# Index entire directory (all files)
docker compose run --rm client index "files/"

# Index only 5 random files from directory
docker compose run --rm client index "files/" --pdf-count 5

# Local development
cd client
export MISTRAL_OCR_API_KEY="your-key"
bun run src/index.ts index "files/" --pdf-count 3
```

**Supported formats**: PDF, PNG, JPG, DOCX, PPTX, TXT, MD

See [Mistral OCR Documentation](https://docs.mistral.ai/capabilities/document_ai/basic_ocr) for details.

#### 3. `search` - Search Indexed Documents

Search previously indexed documents using semantic similarity.

```bash
# Search existing index
docker compose run --rm client search "defamation"

# Local development
bun run src/index.ts search "fraud allegations"
```

#### 4. `run` - Index and Search in One Command

Index a single file and immediately search it.

```bash
# Index and search
docker compose run --rm client run "files/report.pdf" "key findings"

# File outside files/ directory (auto-copies)
docker compose run --rm client run "/tmp/document.pdf" "search term"
```


## Benchmarking

The demo includes comprehensive benchmarking to compare performance across different systems.

### Running Benchmarks

```bash
# Run demo with benchmarking (default: 1 file for quick test)
docker compose run --rm client demo

# Run with more files for accurate benchmarks
docker compose run --rm client demo --pdf-count 5

# Run with all files
docker compose run --rm client demo --pdf-count 0

# Skip saving benchmark (--no-save)
docker compose run --rm client demo --no-save
```

### Benchmark Output

Each demo run outputs:

- **System Information**: Platform (with Docker detection), CPU model, cores, memory, GPU
- **Sample Summary**: Files processed, total size, pages, characters, chunks
- **Timing Summary**: Total duration, OCR time, embedding time (with percentages)
- **OCR Performance**: Average/fastest/slowest times, throughput (chars/sec)
- **Embedding Performance**: Average/fastest/slowest times, chars/sec, tokens/sec
- **Time Projections**: Estimated time per 100/1000 chars, per 1MB/100MB/1GB
- **Text Comparison**: Raw PDF text vs OCR output with similarity percentage
- **Per-File Details**: Hash, size, pages, raw chars, OCR chars, similarity, embed time, chunks

### Comparing Systems

After running benchmarks on multiple machines, compare results:

```bash
# Analyze all sessions in logs/
docker compose run --rm client benchmark

# Or analyze a specific folder
docker compose run --rm client benchmark /path/to/logs
```

The analyzer outputs:

- Session overview with unique systems detected
- Performance comparison table
- Rankings (fastest OCR, fastest embedding, fastest overall)
- Speedup ratios between best and worst performers
- System-specific analysis when multiple systems detected
- Recommendations for optimization

### Session Files

Benchmark sessions are saved to `logs/` with format `YYYYMMDD-HHMMSS.json`:

```json
{
  "sessionId": "20251224-170213",
  "system": {
    "platform": "linux (Docker)",
    "arch": "arm64",
    "cpuModel": "Apple Silicon (via Docker)",
    "cpuCores": 10,
    "totalMemoryGB": 32,
    "gpuAvailable": true,
    "gpuInfo": "Apple Silicon GPU (host)"
  },
  "stats": {
    "totalFiles": 5,
    "totalSizeMB": 25.5,
    "embedAvgCharsPerSecond": 15000
  },
  "files": [
    {
      "fileHash": "7f3153875783d430",
      "fileSizeMB": 3.1,
      "pageCount": 6,
      "rawTextChars": 9034,
      "ocrOutputChars": 8375,
      "textSimilarityPercent": 88.6,
      "ocrDurationMs": 5370,
      "embedDurationMs": 4620,
      "chunkCount": 5
    }
  ]
}
```

File names are replaced with SHA256 hashes for privacy when sharing benchmarks. See `client/benchmark-format.md` for the complete JSON schema.


## Automated Testing

### test-docker-stack.sh

Comprehensive test script with automatic platform detection.

```bash
# Run default demo (auto-detects platform)
./test-docker-stack.sh

# Test with specific file
TEST_FILE="path/to/file.pdf" TEST_QUERY="securities fraud" ./test-docker-stack.sh

# Force GPU profile
PROFILE_OVERRIDE=gpu ./test-docker-stack.sh

# Custom command
TEST_MODE=custom CUSTOM_COMMAND="search 'fraud'" ./test-docker-stack.sh
```

**Environment Variables:**

| Variable | Description | Default |
| --- | --- | --- |
| `PROFILE_OVERRIDE` | Force Docker profile: `default`, `gpu`, or `cuda` | auto-detect |
| `TEST_MODE` | Test mode: `demo`, `single`, or `custom` | `demo` |
| `TEST_FILE` | File to test (for `single` mode) | - |
| `TEST_QUERY` | Search query | `securities fraud` |
| `CUSTOM_COMMAND` | Custom CLI command (for `custom` mode) | - |


## Architecture

### Services

1. **inception-cpu** (Port 8005)
   - Embedding service using ModernBERT
   - Model: `freelawproject/modernbert-embed-base_finetune_512`
   - Generates 768-dimensional vectors

2. **inception-gpu** (Port 8005)
   - GPU-accelerated version
   - Requires NVIDIA CUDA 12.4+
   - Much faster processing

3. **doctor** (Port 5050)
   - Document processing service
   - 18 workers by default

4. **client**
   - Bun/TypeScript CLI
   - Handles OCR, indexing, and search

### Data Flow

```
PDF/Image --> Mistral OCR API --> Markdown
                    |
                    v
         Inception Service --> Vector Embeddings
                    |
                    v
              Local JSON Index
                    |
                    v
         Semantic Search (Cosine Similarity)
```


## Docker Profiles

```bash
# CPU-based (default, M1/M2/M3 Mac compatible)
docker compose up -d inception-cpu

# GPU-accelerated (requires NVIDIA GPU)
docker compose --profile gpu up -d inception-gpu

# Full demo stack (Inception + Doctor + Client)
docker compose --profile demo up -d

# Individual services
docker compose --profile doctor up -d doctor
```


## Development

### Local Setup

1. Start services:

   ```bash
   docker compose up -d inception-cpu
   ```

2. Install client dependencies:

   ```bash
   cd client
   bun install
   ```

3. Set environment variables:

   ```bash
   export MISTRAL_OCR_API_KEY="your-key"
   export INCEPTION_URL="http://localhost:8005"
   ```

4. Run commands:

   ```bash
   bun run src/index.ts demo
   ```

### Test Scripts

Located in `tests/` directory:

- `test-docker-stack.sh` - Full end-to-end test of Docker stack (also available at root)
- `test-mistral-ocr.ts` - Test Mistral OCR API standalone

```bash
# Test full Docker stack (from root)
./test-docker-stack.sh

# Test Mistral OCR API only
bun run tests/test-mistral-ocr.ts

# Test with a specific file
bun run tests/test-mistral-ocr.ts "client/files/sample.pdf"
```


## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Required
MISTRAL_OCR_API_KEY=your-key-here

# Optional
INCEPTION_URL=http://localhost:8005
TRANSFORMER_MODEL_NAME=freelawproject/modernbert-embed-base_finetune_512
MAX_BATCH_SIZE=32
DOCTOR_WORKERS=18
```

### Inception Service

Edit `docker-compose.yml` to change settings:

```yaml
environment:
  - TRANSFORMER_MODEL_NAME=your-model-name
  - MAX_BATCH_SIZE=64
```


## Performance

### OCR Speed (Mistral API)

| Document Size | Processing Time |
| --- | --- |
| Small PDFs (< 5 pages) | 5-10 seconds |
| Medium PDFs (5-20 pages) | 15-30 seconds |
| Large PDFs (20+ pages) | 30+ seconds |

### Embedding Speed (Inception)

| Hardware | Time per Chunk |
| --- | --- |
| CPU (Apple Silicon) | 1-2 seconds |
| GPU (NVIDIA CUDA) | 0.1-0.5 seconds |

### Search Speed

- Instant for < 1,000 chunks
- Less than 1 second for 10,000+ chunks


## Costs

### Mistral OCR

- **Standard**: $2 per 1,000 pages
- **Batch API**: $1 per 1,000 pages (50% discount)

### Inception Service

- **Self-hosted**: Free (requires CPU/GPU hardware)


## Troubleshooting

### Service Not Ready

```bash
# Check services
docker compose ps

# View logs
docker compose logs inception-cpu
docker compose logs doctor

# Restart services
docker compose restart
```

### OCR Failures

**Error**: "Upload failed" or "Invalid file format"

- Verify the file is a valid PDF or image
- Check file size (max 50MB, 1000 pages)
- Ensure the API key is set correctly

### Embedding Errors

**Error**: "Connection refused"

- Ensure the Docker service is running
- Check that port 8005 is available
- Wait 1-2 minutes for model loading

### No Search Results

- Verify `embeddings.json` exists and is not empty
- Re-index documents
- Try different search queries


## File Structure

```
inception-demo/
├── client/                     # TypeScript/Bun CLI
│   ├── files/                  # Document storage (PDFs renamed to SHA256 hashes)
│   │   └── filename-mapping.json  # Original filename mapping
│   ├── ocr/                    # OCR output (.ocr.md files)
│   ├── logs/                   # Benchmark session JSON files
│   ├── src/
│   │   ├── index.ts            # CLI commands
│   │   ├── api.ts              # OCR and embedding functions
│   │   ├── pdf-utils.ts        # PDF text extraction and comparison
│   │   ├── types.ts            # TypeScript type definitions
│   │   ├── benchmark.ts        # Benchmark analyzer
│   │   ├── benchmark-utils.ts  # Benchmark utilities and system detection
│   │   └── rename-to-hash.ts   # Utility to rename files to SHA256 hashes
│   ├── benchmark-format.md     # Benchmark JSON schema documentation
│   └── embeddings.json         # Vector index (generated)
├── inception/                  # Python embedding service
├── doctor/                     # Document processing service
├── tests/                      # Test scripts
│   ├── test-docker-stack.sh    # Full Docker stack test
│   └── test-mistral-ocr.ts     # Mistral OCR API test
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Example environment configuration
├── docker-compose.yml          # Service orchestration
├── start-inception.sh          # Platform-aware startup script
├── test-docker-stack.sh        # Automated testing (symlink to tests/)
├── CHANGELOG.md                # Version history
└── README.md                   # This file
```


## GPU Support

> **Note**: GPU support is currently not working and will be addressed in a future update. Use the CPU version (`inception-cpu`) for now.

### Requirements

- NVIDIA GPU with CUDA 12.6+
- NVIDIA Docker runtime
- Linux host (GPU passthrough from Mac/Windows not supported)

### Usage

```bash
# Build GPU image
docker compose --profile gpu build inception-gpu

# Run GPU service
docker compose --profile gpu up -d inception-gpu

# Test with GPU
PROFILE_OVERRIDE=gpu ./test-docker-stack.sh
```

### Verify GPU Access

```bash
docker compose exec inception-gpu nvidia-smi
```


## CI/CD Integration

The `test-docker-stack.sh` script is designed for CI/CD:

```yaml
# Example GitHub Actions
- name: Run Inception Demo Tests
  env:
    MISTRAL_OCR_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
  run: |
    echo "MISTRAL_OCR_API_KEY=$MISTRAL_OCR_API_KEY" > .env
    ./test-docker-stack.sh
```


## Resources

- **Mistral OCR**: [https://docs.mistral.ai/capabilities/document_ai/basic_ocr](https://docs.mistral.ai/capabilities/document_ai/basic_ocr)
- **Inception Service**: [https://github.com/freelawproject/inception](https://github.com/freelawproject/inception)
- **ModernBERT Model**: [https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512](https://huggingface.co/freelawproject/modernbert-embed-base_finetune_512)
- **Free Law Project**: [https://free.law/2025/03/11/semantic-search/](https://free.law/2025/03/11/semantic-search/)


## License

See individual component licenses:

- Inception: Check [inception/LICENSE](inception/LICENSE)
- Doctor: Check [doctor/LICENSE](doctor/LICENSE)
- Client: MIT License


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `./test-docker-stack.sh`
5. Submit a pull request


## Support

For issues and questions:

- Mistral API: [https://docs.mistral.ai](https://docs.mistral.ai)
- Inception: [https://github.com/freelawproject/inception/issues](https://github.com/freelawproject/inception/issues)
- This Demo: Open an issue in this repository
