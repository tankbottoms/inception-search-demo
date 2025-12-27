# TODO

## vLLM Hydra Client - Planned Features

### High Priority

- [ ] **Streaming Responses**: Stream long inference responses
- [ ] **Batch PDF Processing**: Process multiple PDFs in parallel
- [ ] **Progress Indicators**: Show progress for long-running operations

### Medium Priority

- [ ] **Vector Database Integration**: ChromaDB/Milvus instead of in-memory index
- [ ] **RAG Pipeline**: Retrieval-Augmented Generation with indexed docs
- [ ] **API Server Mode**: Expose client as REST API
- [ ] **Document Chunking**: Smart chunking for large documents

### Low Priority

- [ ] **Web UI**: Browser-based interface
- [ ] **Export Formats**: PDF, DOCX output
- [ ] **OCR Confidence Scoring**: Quality metrics for OCR results
- [ ] **Parallel Page OCR**: Process pages concurrently
- [ ] **Embedding Cache**: Avoid recomputation of embeddings

### Ideas

- Interactive chat mode
- A/B testing for prompts
- Legal database integration for citation verification
- Comparison mode for different OCR providers

---

## Completed

### v2.2.0 (2025-12-27)

- [x] **Bandwidth Benchmark**: Multi-concurrency throughput testing
- [x] **Load Balancer Support**: Auto-detect and compare LB endpoint
- [x] **Head-to-Head Comparison**: Single vs LB with improvement %
- [x] **Batch Embedding Tests**: Test batch sizes 1, 5, 10, 20
- [x] **Rich JSON Reports**: Full config and results in output
- [x] **New CLI Options**: --concurrency, --requests, --skip-lb

### v2.1.0 (2025-12-27)

- [x] **Full Demo Workflow**: OCR -> Embed -> Search -> Citation
- [x] **GPU Monitoring**: Real-time utilization display
- [x] **Benchmark Statistics**: Per-operation timing and GPU stats
- [x] **imgcat Support**: Display images in iTerm2

### v2.0.0 (2025-12-26)

- [x] OCR command with PDF and image support
- [x] Pipeline command (PDF -> OCR -> Embed -> Search)
- [x] PDF processing with pdf2pic and sharp
- [x] Text cleaning for embeddings

### v1.0.0 (2025-12-25)

- [x] Basic client with health, test, demo, benchmark commands
- [x] Dynamic model ID fetching from /v1/models
- [x] PDF to image conversion for OCR
- [x] Multi-page OCR processing
- [x] Embeddings generation with ModernBERT
- [x] Semantic search with cosine similarity
- [x] Color-coded output

---

## Notes

### Benchmark Results Summary

| Test | Result |
|------|--------|
| Single Request Latency | ~9ms |
| Single Replica Throughput | ~320 req/s @ 20 concurrent |
| Load Balanced Throughput | ~645 req/s @ 20 concurrent |
| LB Improvement | ~100% at high concurrency |
| Batch (20 docs) | 850 docs/s |

### Performance Tips

1. Use batch API for highest throughput (8x improvement)
2. Enable load balancing for concurrent workloads
3. Skip OCR in benchmarks unless testing OCR specifically
4. Use --skip-lb flag when LB is not configured
