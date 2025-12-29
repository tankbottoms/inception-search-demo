# TODO

Implementation tasks for the ONNX TypeScript backend.

## Phase 2: Core Backend ✅ COMPLETE

- [x] Set up Hono server in `src/index.ts`
- [x] Implement provider detection in `src/services/hardware.ts`
- [x] Create model registry loader in `src/services/model-loader.ts`
- [x] Build settings/config management in `src/services/config.ts`
- [x] Add structured logging in `src/services/logger.ts`

## Phase 3: Python Converter ✅ COMPLETE

- [x] Implement CLI conversion in `converter/convert.py`
- [x] Add HTTP API in `converter/server.py`
- [x] Extract pooling config from SentenceTransformers models
- [x] Create `converter/Dockerfile` with optimum, torch, transformers

## Phase 4: Inference Services ✅ COMPLETE

- [x] Implement tokenization using custom sentence tokenizer in `src/services/tokenizer.ts`
- [x] Create ONNX session management in `src/services/model-loader.ts`
- [x] Implement mean pooling + L2 normalization in `src/services/embedding.ts`
- [x] Port text chunking logic in `src/services/tokenizer.ts`

## Phase 5: OCR Integration ✅ COMPLETE

- [x] Port Mistral OCR client to `src/services/ocr/mistral.ts`
- [x] Create unified OCR interface in `src/services/ocr/index.ts`
- [x] Create HunyuanOCR Python->ONNX converter in `converter/convert_hunyuan_ocr.py`
- [x] Implement HunyuanOCR ONNX inference in `src/services/ocr/hunyuan.ts`
- [x] Add image preprocessing pipeline with sharp
- [x] Support both CPU and CUDA ONNX runtime providers
- [x] Remove DeepSeek-OCR (not compatible with ONNX)

## Phase 6: Demo Client ✅ COMPLETE

- [x] Create demo CLI in `demo/src/index.ts`
- [x] Implement PDF processing pipeline (text extraction + OCR fallback)
- [x] Build OCR -> Embedding workflow
- [x] Add search with cosine similarity
- [x] Implement benchmark reporting
- [x] Add Makefile with simple commands

## Phase 7: Docker & Scripts

- [x] Create `Dockerfile` for CPU builds
- [x] Create `Dockerfile.cuda` for GPU builds
- [x] Update `docker-compose.yml` with new profiles
- [x] Create `scripts/startup.sh` with auto-detection
- [ ] Add `scripts/check-models.sh` for model validation
- [ ] Add `scripts/benchmark.sh` for comparison runs

## Phase 9: Benchmarking ✅ COMPLETE

- [x] Implement CLI benchmarking in `src/cli.ts`
- [x] Implement CPU vs GPU comparison
- [x] Generate comparison reports
- [x] Document benchmark methodology

### Benchmark Results (2025-12-29)

| Metric | TypeScript (CPU) | Python (GPU) | Speedup |
|--------|------------------|--------------|---------|
| Avg Latency | 80ms | 30ms | **2.7x** |
| P95 Latency | 137ms | 39ms | **3.5x** |
| Throughput | 12.47 req/s | 33.05 req/s | **2.6x** |
| Batch (10 docs) | 23.44 docs/s | 246.30 docs/s | **10.5x** |

See `benchmark-results/benchmark-2025-12-29.md` for full report.

## Validation

- [ ] Verify embedding outputs match Python backend (within epsilon)
- [ ] Test on Apple Silicon (M1-M4)
- [x] Test on DGX Spark (ARM64 + CUDA)
- [x] Performance benchmarks vs Python baseline
- [x] Removed legacy `backup/` folder
