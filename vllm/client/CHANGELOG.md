# Changelog

All notable changes to the vLLM Hydra Client.

## [2.2.0] - 2025-12-27

### Added

- **Enhanced Bandwidth Benchmark**
  - Latency benchmark with min/max/p95 statistics
  - Throughput benchmark at multiple concurrency levels
  - Single replica vs load-balanced endpoint comparison
  - Head-to-head comparison with improvement percentage
  - Batch embedding tests (1, 5, 10, 20 documents)

- **Rich JSON Reports**
  - Full benchmark configuration in output
  - Service discovery status and model info
  - Detailed results per test
  - Summary with peak throughput and LB improvement

- **New CLI Options**
  - `--concurrency <levels>`: Comma-separated concurrency levels (default: "1,5,10,20")
  - `--requests <n>`: Total requests for throughput test (default: 50)
  - `--skip-lb`: Skip load balancer comparison

- **Load Balancer Support**
  - Auto-detect load-balanced endpoint at port 8000
  - Environment variable `LB_URL` for custom endpoint
  - Comparison benchmarks when LB is available

### Changed

- Benchmark command now shows service discovery status
- Improved benchmark output formatting with sections
- Better error handling for unavailable services

## [2.1.0] - 2025-12-27

### Added

- **Comprehensive Demo** (`demo:full`) - Full OCR-to-citation workflow
  - PDF to image conversion
  - Multi-page OCR using HunyuanOCR
  - Direct PDF text extraction
  - OCR vs extracted text comparison
  - Merged markdown generation using GPT-OSS
  - Semantic search with embeddings
  - Legal citation generation using GPT-OSS
  - GPU utilization monitoring with color-coded output
  - Detailed benchmark statistics

- **GPU Monitoring**
  - Real-time GPU utilization display
  - Memory usage tracking
  - Per-operation GPU stats in benchmarks

- **Status Command** (`demo:status`)
  - Shows GPU utilization, memory, temperature
  - Service availability check

### Changed

- Updated service ports: OCR moved to 8003 (HunyuanOCR)
- Improved text cleaning for embeddings
- Dynamic model ID fetching from /v1/models

## [2.0.0] - 2025-12-26

### Added

- OCR command with PDF and image support
- Pipeline command (PDF -> OCR -> Embed -> Search)
- PDF processing with pdf2pic and sharp
- Text cleaning to handle special characters

### Changed

- Fixed model ID handling (was using "default", now fetches actual ID)
- Updated service URLs to match docker-compose

## [1.0.0] - 2025-12-25

### Added

- Initial release
- Health check command
- Test command for all services
- Demo command with embeddings and similarity search
- Benchmark command with performance metrics

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 2.2.0 | 2025-12-27 | Bandwidth benchmark, LB comparison, rich reports |
| 2.1.0 | 2025-12-27 | Full demo workflow, GPU monitoring |
| 2.0.0 | 2025-12-26 | OCR, pipeline, PDF processing |
| 1.0.0 | 2025-12-25 | Initial release |
