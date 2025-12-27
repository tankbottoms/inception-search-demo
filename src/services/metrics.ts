/**
 * Prometheus metrics for Inception ONNX service
 * Mirrors Python metrics from backup/inception/inception/metrics.py
 */

import { Counter, Histogram, Gauge, Registry, collectDefaultMetrics } from 'prom-client';
import { settings } from './config';

// Create a new registry
export const registry = new Registry();

// Collect default metrics (CPU, memory, etc.)
if (settings.enableMetrics) {
  collectDefaultMetrics({ register: registry });
}

// ============================================================
// Custom Metrics
// ============================================================

/**
 * Request counter by endpoint and status
 */
export const requestCount = new Counter({
  name: 'inception_request_total',
  help: 'Total number of requests by endpoint and status',
  labelNames: ['endpoint', 'status'],
  registers: [registry],
});

/**
 * Processing time histogram by endpoint
 */
export const processingTime = new Histogram({
  name: 'inception_processing_seconds',
  help: 'Time spent processing requests in seconds',
  labelNames: ['endpoint'],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [registry],
});

/**
 * Error counter by endpoint and error type
 */
export const errorCount = new Counter({
  name: 'inception_errors_total',
  help: 'Total number of errors by endpoint and type',
  labelNames: ['endpoint', 'error_type'],
  registers: [registry],
});

/**
 * Chunk count histogram by endpoint
 */
export const chunkCount = new Counter({
  name: 'inception_chunks_total',
  help: 'Total number of chunks processed by endpoint',
  labelNames: ['endpoint'],
  registers: [registry],
});

/**
 * Model load time histogram
 */
export const modelLoadTime = new Histogram({
  name: 'inception_model_load_seconds',
  help: 'Time spent loading models in seconds',
  buckets: [0.5, 1, 2.5, 5, 10, 30, 60],
  registers: [registry],
});

/**
 * Active inference sessions gauge
 */
export const activeSessions = new Gauge({
  name: 'inception_active_sessions',
  help: 'Number of active inference sessions',
  registers: [registry],
});

/**
 * GPU memory usage gauge (if available)
 */
export const gpuMemoryUsed = new Gauge({
  name: 'inception_gpu_memory_bytes',
  help: 'GPU memory used in bytes',
  labelNames: ['type'],
  registers: [registry],
});

/**
 * Batch size histogram
 */
export const batchSizeHistogram = new Histogram({
  name: 'inception_batch_size',
  help: 'Distribution of batch sizes',
  buckets: [1, 5, 10, 25, 50, 100],
  registers: [registry],
});

/**
 * Embedding dimension gauge
 */
export const embeddingDimension = new Gauge({
  name: 'inception_embedding_dimension',
  help: 'Embedding dimension of the loaded model',
  registers: [registry],
});

// ============================================================
// Metrics Helper Object
// ============================================================

export const metrics = {
  requestCount,
  processingTime,
  errorCount,
  chunkCount,
  modelLoadTime,
  activeSessions,
  gpuMemoryUsed,
  batchSizeHistogram,
  embeddingDimension,

  /**
   * Record an error
   */
  recordError(endpoint: string, errorType: string): void {
    errorCount.inc({ endpoint, error_type: errorType });
    requestCount.inc({ endpoint, status: 'error' });
  },

  /**
   * Get metrics as Prometheus text format
   */
  async getMetrics(): Promise<string> {
    return registry.metrics();
  },

  /**
   * Get metrics content type
   */
  getContentType(): string {
    return registry.contentType;
  },

  /**
   * Reset all metrics (for testing)
   */
  reset(): void {
    registry.resetMetrics();
  },
};
