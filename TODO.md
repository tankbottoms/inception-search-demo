# TODO

Implementation tasks for the ONNX TypeScript backend.

## Phase 2: Core Backend

- [ ] Set up Hono server in `src/index.ts`
- [ ] Implement provider detection in `src/providers/provider-factory.ts`
- [ ] Create model registry loader in `src/services/model-registry.ts`
- [ ] Build HuggingFace API client in `src/services/huggingface.ts`
- [ ] Add structured logging in `src/instrumentation/logger.ts`

## Phase 3: Python Converter

- [ ] Create `converter/Dockerfile` with optimum, torch, transformers
- [ ] Implement CLI conversion in `converter/convert.py`
- [ ] Add HTTP API in `converter/server.py`
- [ ] Extract pooling config from SentenceTransformers models

## Phase 4: Inference Services

- [ ] Implement tokenization using Transformers.js
- [ ] Create ONNX session management in `src/services/model-loader.ts`
- [ ] Implement mean pooling + L2 normalization in `src/services/pooling.ts`
- [ ] Port text chunking logic from Python in `src/services/chunking.ts`

## Phase 5: OCR Integration

- [ ] Port Mistral OCR client to `src/services/ocr/mistral.ts`
- [ ] Implement DeepSeek-OCR local inference
- [ ] Implement HunyuanOCR local inference
- [ ] Create unified OCR route in `src/routes/ocr.ts`

## Phase 6: Demo Client

- [ ] Create demo CLI in `demo/src/index.ts`
- [ ] Implement PDF processing pipeline
- [ ] Build OCR -> Embedding workflow
- [ ] Add search with cosine similarity
- [ ] Implement benchmark reporting

## Phase 7: Docker & Scripts

- [ ] Create `Dockerfile` for CPU builds
- [ ] Create `Dockerfile.cuda` for GPU builds
- [ ] Update `docker-compose.yml` with new profiles
- [ ] Create `scripts/startup.sh` with auto-detection
- [ ] Add `scripts/check-models.sh` for model validation
- [ ] Add `scripts/benchmark.sh` for comparison runs

## Phase 8: vLLM Alternative

- [ ] Create `vllm/docker-compose.spark-1.yml`
- [ ] Create `vllm/docker-compose.spark-2.yml`
- [ ] Implement `vllm/scripts/start-cluster.sh`
- [ ] Implement `vllm/scripts/stop-cluster.sh`
- [ ] Add `vllm/scripts/monitor.sh` for crontab monitoring
- [ ] Document vLLM setup in `vllm/README.md`

## Phase 9: Benchmarking

- [ ] Implement CPU vs GPU comparison
- [ ] Add vLLM distributed benchmarks
- [ ] Generate comparison reports
- [ ] Document benchmark methodology

## Validation

- [ ] Verify embedding outputs match Python backend (within epsilon)
- [ ] Test on Apple Silicon (M1-M4)
- [ ] Test on DGX Spark (ARM64 + CUDA)
- [ ] Performance benchmarks vs Python baseline
- [ ] Delete `backup/` folder after validation
