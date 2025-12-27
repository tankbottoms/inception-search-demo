# Changelog

All notable changes to the vLLM Hydra Cluster will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-12-27

### Added

- **Makefile** - Comprehensive command interface for all operations
  - `make up-all` - Start all services with one command
  - `make test` - Run full verification suite (14 tests)
  - `make demo-cot` - Chain-of-thought reasoning demos
  - `make stress` - Load balancer stress testing
  - Multi-node deployment commands (`make deploy`, `make workers`)

- **Chain-of-Thought Demo** (`cot-demo.ts`)
  - GPT-OSS 20B reasoning visualization
  - 4 example categories: Math, Logic, Code, Legal
  - Verification tests with expected answers
  - ~44 tokens/sec throughput

- **OCR Pipeline Demo** (`ocr-pipeline.ts`)
  - Full PDF processing: OCR → Text merge → Summary → Citation
  - Blue Book legal citation generation
  - GPU benchmarking and verification
  - Markdown report generation

- **Verification Suite** (`verify-demo.ts`)
  - 14 automated tests across OCR, Embeddings, and Search
  - OCR text extraction accuracy testing (5 tests)
  - Semantic similarity verification (5 tests)
  - Search ranking validation (4 tests)

- **Stress Test** (`stress-test.ts`)
  - Embedding load balancer throughput testing
  - Configurable concurrency and duration
  - Single vs load-balanced comparison
  - Latency percentiles (p50, p95, p99)
  - 640+ req/s single, 1400+ req/s load balanced

### Changed

- Updated README with Makefile-based quick start
- Added multi-node deployment documentation (spark-1 + spark-2)
- Improved test output formatting (benchmark style)
- Updated package.json to v2.2.0 with new scripts

### Fixed

- GPT-OSS HarmonyError with vocab file loading (requires vLLM 25.12+)
- Chain-of-thought demo legal reasoning completing properly
- OCR verification now uses generated test images

## [1.2.0] - 2025-12-27

### Added

- **Load Balancing with Traefik**
  - `embeddings-scaled` profile for 2x embedding replicas
  - Traefik load balancer with round-robin distribution
  - Unified endpoint at `http://localhost:8000/v1/embeddings`
  - Health checks with automatic failover
  - Dashboard at `http://localhost:8088`
  - Configuration in `traefik/traefik.yml` and `traefik/dynamic.yml`

- **Enhanced Bandwidth Benchmark**
  - Latency benchmarks with min/max/p95 statistics
  - Throughput testing at multiple concurrency levels (1, 5, 10, 20)
  - Single vs load-balanced endpoint comparison
  - Head-to-head comparison with improvement percentage
  - Batch embedding tests (1, 5, 10, 20 docs)
  - Rich JSON report with full configuration and results

- **GPT-OSS Loading Optimizations**
  - Persistent `torch.compile` cache volume (`vllm-compile-cache`)
  - Reduced `max-num-seqs` for faster KV cache warmup
  - Cold start: ~165s, Warm start: ~100s (was ~195s)
  - New `gpt-oss-eager` profile for fastest startup (~80s, slower inference)

- **Client CLI Improvements**
  - New benchmark options: `--concurrency`, `--requests`, `--skip-lb`
  - Service discovery for single replica and load-balanced endpoints
  - Environment variable `LB_URL` for load balancer endpoint

### Changed

- Updated README with comprehensive QUICKSTART section
- Added benchmark results table to documentation
- Reorganized architecture diagram with load balancing

### Fixed

- Port conflict resolution (Traefik dashboard moved from 8080 to 8088)

## [1.1.0] - 2025-12-26

### Added

- **Tencent HunyuanOCR Integration**
  - Replaced DeepSeek-OCR with Tencent HunyuanOCR
  - Better OCR accuracy for legal documents
  - Port 8003 for OCR service

- **GPT-OSS 20B Service**
  - Large language model for general inference
  - Port 8004 with `gpt-oss` profile
  - 45% GPU memory allocation

- **Llama 3.1 8B Alternative**
  - Faster loading option (~30s vs ~165s)
  - Same port 8004 with `llama` profile

- **Demo Client Enhancements**
  - Full OCR-to-citation workflow (`demo:full`)
  - GPU utilization monitoring
  - Benchmark statistics collection
  - imgcat support for iTerm2

### Changed

- Updated service port assignments (OCR: 8002 -> 8003)
- Improved GPU memory allocation defaults

## [1.0.0] - 2025-12-26

### Added

- **vLLM Hydra Stack**: Multi-model inference with single Docker Compose
  - `vllm-freelaw-modernbert`: Embeddings service (FreeLaw ModernBERT)
  - Core architecture for multiple vLLM services

- **Monitor Service**: Cron-based health monitoring with auto-restart
  - Docker container (`hydra-monitor`) for containerized monitoring
  - Local script with `--watch` mode for continuous monitoring
  - Crontab installation with `--monitor` flag
  - Manual shutdown detection to prevent restart conflicts

- **Scripts**:
  - `hydra-up.sh`: Start services with options
  - `hydra-down.sh`: Stop services with manual shutdown flag
  - `hydra-monitor.sh`: Health monitoring with multiple modes
  - `test-hydra-stack.sh`: Automated test suite with benchmark option

- **Client Application**: TypeScript client for demo and benchmarking
  - `demo` command: Full pipeline demonstration
  - `benchmark` command: Performance benchmarks
  - `test` command: Service tests
  - `health` command: Health check

- **Configuration**:
  - `.env` and `.env.example` with comprehensive settings
  - GPU memory allocation per service
  - Model selection and port configuration

- **Documentation**:
  - Comprehensive README with QUICKSTART guide
  - Docker Compose command reference
  - API usage examples
  - Troubleshooting guide

## [0.1.0] - 2025-12-25

### Added

- Initial distributed inference setup for DGX Spark nodes
- Ray cluster configuration for tensor parallelism
- Basic monitoring script
- Support for HunyuanOCR and other vLLM-compatible models

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 2.2.0 | 2025-12-27 | Makefile, CoT demo, verification suite, stress test |
| 1.2.0 | 2025-12-27 | Load balancing, bandwidth benchmarks, GPT-OSS optimizations |
| 1.1.0 | 2025-12-26 | HunyuanOCR, GPT-OSS, Llama, demo enhancements |
| 1.0.0 | 2025-12-26 | vLLM Hydra multi-model stack |
| 0.1.0 | 2025-12-25 | Initial distributed inference setup |
