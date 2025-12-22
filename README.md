# Inception Demo - Document OCR and Semantic Search

A complete document processing and semantic search system combining **Mistral OCR**, **Free Law Project Inception embeddings**, and **vector similarity search**.

## Overview

This demo showcases:

- **Mistral OCR API** - Converts PDFs and images to markdown ($2 per 1,000 pages)
- **Inception Embedding Service** - Generates vector embeddings using ModernBERT
- **Semantic Search** - Find relevant content using cosine similarity
- **Doctor Service** - Additional document processing capabilities
- **CLI Tools** - Easy-to-use command-line interface

## Features

- Multi-format OCR: PDFs, images, Office documents
- Semantic Search: Context-aware document retrieval
- GPU Support: CUDA 12.6+ acceleration for faster processing
- Docker-based: Complete containerized stack
- Automated Testing: Built-in test automation

## Quickstart

### Prerequisites

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- [Bun](https://bun.sh/) (for running tests)
- Mistral API Key (get one at [https://console.mistral.ai](https://console.mistral.ai))

### 1. Configure Environment

Create `.env` in the project root:

```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```

Or create it manually:

```bash
echo "MISTRAL_OCR_API_KEY=your-api-key-here" > .env
```

### 2. Run the Demo

The test script validates everything works end-to-end:

```bash
# CPU mode (default)
./tests/test-docker-stack.sh

# GPU mode (requires NVIDIA GPU with CUDA 12.6+)
PROFILE=gpu ./tests/test-docker-stack.sh
```

The script will:
1. **Test Mistral OCR API** - Validates your API key works
2. **Build Docker images** - Creates inception-cpu/gpu and client images
3. **Start services** - Launches the embedding service
4. **Run demo** - Index sample files and search for "securities fraud"
5. **Show results** - Display search results with similarity scores
6. **Clean up** - Stop and remove containers

### 3. Quick Test (API Only)

To test just the Mistral OCR API without Docker:

```bash
cd client && bun install && cd ..
bun run tests/test-mistral-ocr.ts
```

## Usage

### Command Reference

The CLI provides four main commands:

#### 1. `demo` - Run Full Demo

Index all files in `files/` directory and perform a semantic search.

```bash
# Default: search for "securities fraud" in all files
docker compose run --rm client demo

# Add a new file and use custom query
docker compose run --rm client demo /path/to/document.pdf "custom query"

# Local development
cd client
bun run src/index.ts demo
```

#### 2. `index` - Index Documents

Index a single file or directory of documents.

```bash
# Index a single PDF
docker compose run --rm client index "files/document.pdf"

# Index entire directory
docker compose run --rm client index "files/"

# Local development
cd client
export MISTRAL_OCR_API_KEY="your-key"
bun run src/index.ts index "files/"
```

**Supported formats**: PDF, PNG, JPG, DOCX, PPTX, TXT, MD

#### 3. `search` - Search Indexed Documents

Search previously indexed documents.

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

## Automated Testing

### test-docker-stack.sh

Comprehensive test script with multiple modes:

```bash
# Run default demo
./test-docker-stack.sh

# Test with specific file
TEST_FILE="path/to/file.pdf" TEST_QUERY="securities fraud" ./test-docker-stack.sh

# Use GPU profile
PROFILE=gpu ./test-docker-stack.sh

# Custom command
TEST_MODE=custom CUSTOM_COMMAND="search 'fraud'" ./test-docker-stack.sh
```

**Environment Variables:**
- `PROFILE` - Docker profile: `default`, `gpu`, or `demo` (default: `default`)
- `TEST_MODE` - Test mode: `demo`, `single`, or `custom` (default: `demo`)
- `TEST_FILE` - File to test (for `single` mode)
- `TEST_QUERY` - Search query (default: "securities fraud")
- `CUSTOM_COMMAND` - Custom CLI command (for `custom` mode)

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
                   PDF/Image → Mistral OCR API → Markdown
                                       ↓
                    Inception Service → Vector Embeddings
                                       ↓
                                 Local JSON Index
                                       ↓
                     Semantic Search (Cosine Similarity)
```

## Docker Profiles

```bash
# CPU-based (default, M1/M2 Mac compatible)
docker compose up -d

# GPU-accelerated (requires NVIDIA GPU)
docker compose --profile gpu up -d

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

- `test-docker-stack.sh` - Full end-to-end test of Docker stack
- `test-mistral-ocr.ts` - Test Mistral OCR API standalone

```bash
# Test full Docker stack
./tests/test-docker-stack.sh

# Test Mistral OCR API only
bun run tests/test-mistral-ocr.ts

# Test with a specific file
bun run tests/test-mistral-ocr.ts "client/files/sample.pdf"
```

## Configuration

### Environment Variables

**Project Root (`.env`):**
```bash
# Required - Mistral OCR API key
MISTRAL_OCR_API_KEY=your-key-here
```

**Docker Compose:**
- `TRANSFORMER_MODEL_NAME` - Embedding model (default: modernbert)
- `MAX_BATCH_SIZE` - Batch size for embeddings (default: 32)
- `DOCTOR_WORKERS` - Number of doctor workers (default: 18)

### Inception Service

Edit `docker-compose.yml` to change:

```yaml
environment:
  - TRANSFORMER_MODEL_NAME=your-model-name
  - MAX_BATCH_SIZE=64
```

## Performance

### OCR Speed (Mistral API)
- Small PDFs (< 5 pages): 5-10 seconds
- Medium PDFs (5-20 pages): 15-30 seconds
- Large PDFs (20+ pages): 30+ seconds

### Embedding Speed (Inception)
- CPU: 1-2 seconds per chunk
- GPU: 0.1-0.5 seconds per chunk

### Search Speed
- Instant for < 1,000 chunks
- < 1 second for 10,000+ chunks

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
- Verify file is valid PDF/image
- Check file size (max 50MB, 1000 pages)
- Ensure API key is set correctly

### Embedding Errors

**Error**: "Connection refused"
- Ensure Docker service is running
- Check port 8005 is available
- Wait 1-2 minutes for model loading

### No Search Results

- Verify `embeddings.json` exists and is not empty
- Re-index documents
- Try different search queries

## File Structure

```
inception-demo/
├── .env                   # Environment variables (API keys) - create from .env.example
├── .env.example           # Example environment file
├── client/                # TypeScript/Bun CLI
│   ├── files/             # Document storage
│   ├── src/
│   │   ├── index.ts       # CLI commands
│   │   └── api.ts         # OCR and embedding functions
│   └── embeddings.json    # Vector index (generated)
├── inception/             # Python embedding service
├── doctor/                # Document processing service
├── tests/                 # Test scripts
│   ├── test-docker-stack.sh    # Full Docker stack test
│   └── test-mistral-ocr.ts     # Mistral OCR API test
├── docker-compose.yml     # Service orchestration
└── README.md              # This file
```

## GPU Support

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
PROFILE=gpu ./tests/test-docker-stack.sh
```

### Verify GPU Access

```bash
docker compose exec inception-gpu nvidia-smi
```

## CI/CD Integration

The `tests/test-docker-stack.sh` script is designed for CI/CD:

```yaml
# Example GitHub Actions
- name: Run Inception Demo Tests
  env:
    MISTRAL_OCR_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
  run: |
    echo "MISTRAL_OCR_API_KEY=$MISTRAL_OCR_API_KEY" > .env
    ./tests/test-docker-stack.sh
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
