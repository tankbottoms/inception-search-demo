# CPU vs GPU Benchmark Results

**Date**: 2025-12-29
**Platform**: NVIDIA DGX Spark (ARM64 + CUDA)

## System Configuration

| Component | Details |
|-----------|---------|
| Platform | Linux 6.14.0-1015-nvidia (ARM64) |
| GPU | NVIDIA GB10 (Grace Hopper) |
| CUDA | 13.0 |
| Model | freelawproject/modernbert-embed-base_finetune_512 |
| Embedding Dim | 768 |

## Test Configuration

| Setting | Value |
|---------|-------|
| Warmup Iterations | 5 |
| Benchmark Iterations | 20 |
| Text Samples | 3 (short, medium, long) |
| Batch Size | 10 documents |

## Services Tested

| Service | Port | Provider | Backend |
|---------|------|----------|---------|
| TypeScript/ONNX | 8005 | CPU (fallback) | onnxruntime-node |
| Python/ONNX | 8006 | CUDA (GPU) | onnxruntime-gpu |

## Single Query Results

| Metric | TypeScript (CPU) | Python (GPU) | Speedup |
|--------|------------------|--------------|---------|
| **Avg Latency** | 80ms | 30ms | **2.7x** |
| Min Latency | 36ms | 22ms | 1.6x |
| Max Latency | 148ms | 54ms | 2.7x |
| P50 Latency | 68ms | 30ms | 2.3x |
| P95 Latency | 137ms | 39ms | 3.5x |
| P99 Latency | 148ms | 54ms | 2.7x |
| **Throughput** | 12.47 req/s | 33.05 req/s | **2.6x** |

## Batch Processing Results

| Metric | TypeScript (CPU) | Python (GPU) | Speedup |
|--------|------------------|--------------|---------|
| Batch Size | 10 docs | 10 docs | - |
| Avg Batch Latency | 426ms | 40ms | **10.7x** |
| **Docs/second** | 23.44 | 246.30 | **10.5x** |

## Key Findings

1. **GPU Acceleration**: The Python GPU backend with ONNX Runtime CUDA is significantly faster:
   - **2.7x faster** for single queries
   - **10.5x faster** for batch processing

2. **Batch Efficiency**: GPU shows dramatic improvement in batch processing due to parallel execution on CUDA cores.

3. **Latency Consistency**: GPU backend shows more consistent latency (P95=39ms vs P95=137ms for CPU).

4. **Throughput**: GPU achieves 33 req/s vs CPU's 12.5 req/s for single queries.

## Architecture Notes

- **TypeScript Backend**: Uses `onnxruntime-node` which only supports CPU execution. The CUDA provider is not available in this package.

- **Python Backend**: Uses `onnxruntime-gpu` with full CUDA support on the NVIDIA GB10 GPU.

- **Recommendation**: For production deployments on GPU-enabled systems, use the Python backend. For CPU-only systems (Apple Silicon, generic ARM64), the TypeScript backend provides acceptable performance.

## Next Steps

- [ ] Test with larger batch sizes (32, 64, 128)
- [ ] Compare memory usage between backends
- [ ] Test OCR performance (HunyuanOCR)
- [ ] Implement TypeScript CUDA support via onnxruntime-gpu npm package
