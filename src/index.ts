/**
 * Inception ONNX - TypeScript/Bun Inference Backend
 *
 * Multi-platform ONNX inference service with ARM64 CPU and CUDA GPU acceleration
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger as honoLogger } from 'hono/logger';
import type { Context } from 'hono';
import type {
  QueryRequest,
  QueryResponse,
  TextRequest,
  BatchTextRequest,
  TextResponse,
} from './types';
import {
  logger,
  settings,
  metrics,
  getHardwareStatus,
  initEmbeddingService,
  generateQueryEmbedding,
  generateTextEmbedding,
  generateBatchEmbeddings,
  getServiceStatus,
  loadRegistry,
  performOCR,
  performBatchOCR,
  getAvailableProviders,
  getProviderStatus,
  initializeOCR,
  isMistralConfigured,
  isHunyuanAvailable,
} from './services';

const app = new Hono();

// ============================================================
// Middleware
// ============================================================

app.use('*', cors());
app.use('*', honoLogger());

// Error handler
app.onError((err, c) => {
  logger.error('Request error', err);
  metrics.recordError(c.req.path, err.name || 'UnknownError');

  const status = (err as any).statusCode || 500;
  return c.json(
    {
      error: err.message,
      code: (err as any).code || 'INTERNAL_ERROR',
    },
    status
  );
});

// ============================================================
// Health & Status Endpoints
// ============================================================

app.get('/health', (c) => {
  const hw = getHardwareStatus();
  const service = getServiceStatus();

  return c.json({
    status: 'ok',
    version: '2.0.0',
    provider: hw.provider,
    device: hw.deviceName,
    model: service.modelId || 'not loaded',
    initialized: service.initialized,
    timestamp: new Date().toISOString(),
  });
});

app.get('/status', (c) => {
  const hw = getHardwareStatus();
  const service = getServiceStatus();
  const registry = loadRegistry();

  return c.json({
    service: {
      version: '2.0.0',
      initialized: service.initialized,
      modelId: service.modelId,
      embeddingDim: service.embeddingDim,
    },
    hardware: {
      provider: hw.provider,
      device: hw.deviceName,
      cudaVersion: hw.cudaVersion,
      memory: hw.memoryTotal
        ? {
            total_mb: hw.memoryTotal,
            free_mb: hw.memoryFree,
            used_mb: (hw as any).memoryUsed,
          }
        : undefined,
    },
    models: registry.models.filter(m => m.enabled),
    config: {
      maxTokens: settings.maxTokens,
      maxBatchSize: settings.maxBatchSize,
      processingBatchSize: settings.processingBatchSize,
    },
  });
});

app.get('/metrics', async (c) => {
  const metricsText = await metrics.getMetrics();
  return c.text(metricsText, 200, {
    'Content-Type': metrics.getContentType(),
  });
});

// ============================================================
// Embedding Endpoints
// ============================================================

/**
 * POST /api/v1/embed/query
 * Generate embedding for a search query
 */
app.post('/api/v1/embed/query', async (c: Context) => {
  const body = await c.req.json<QueryRequest>();

  if (!body.text) {
    return c.json({ error: 'text is required' }, 400);
  }

  const result = await generateQueryEmbedding(body.text);

  const response: QueryResponse = {
    embedding: result.embedding,
  };

  return c.json(response, 200, {
    'X-Processing-Time-Ms': result.timing.total_ms.toFixed(2),
  });
});

/**
 * POST /api/v1/embed/text
 * Generate embeddings for a single document (with chunking)
 */
app.post('/api/v1/embed/text', async (c: Context) => {
  const body = await c.req.json<TextRequest>();

  if (body.id === undefined || body.id === null) {
    return c.json({ error: 'id is required' }, 400);
  }

  if (!body.text) {
    return c.json({ error: 'text is required' }, 400);
  }

  const result = await generateTextEmbedding(body.id, body.text);

  return c.json(result.response, 200, {
    'X-Processing-Time-Ms': result.timing.total_ms.toFixed(2),
    'X-Chunk-Count': String(result.response.embeddings.length),
  });
});

/**
 * POST /api/v1/embed/batch
 * Generate embeddings for multiple documents
 */
app.post('/api/v1/embed/batch', async (c: Context) => {
  const body = await c.req.json<BatchTextRequest>();

  if (!body.documents || !Array.isArray(body.documents)) {
    return c.json({ error: 'documents array is required' }, 400);
  }

  if (body.documents.length === 0) {
    return c.json({ error: 'documents array cannot be empty' }, 400);
  }

  if (body.documents.length > settings.maxBatchSize) {
    return c.json(
      { error: `batch size ${body.documents.length} exceeds maximum of ${settings.maxBatchSize}` },
      400
    );
  }

  // Validate all documents
  for (const doc of body.documents) {
    if (doc.id === undefined || doc.id === null) {
      return c.json({ error: 'each document must have an id' }, 400);
    }
    if (!doc.text) {
      return c.json({ error: `document ${doc.id} is missing text` }, 400);
    }
  }

  const result = await generateBatchEmbeddings(body.documents);

  const totalChunks = result.results.reduce((sum, r) => sum + r.embeddings.length, 0);

  return c.json(
    {
      results: result.results,
      timing: result.timing,
    },
    200,
    {
      'X-Processing-Time-Ms': result.timing.total_ms.toFixed(2),
      'X-Document-Count': String(result.results.length),
      'X-Chunk-Count': String(totalChunks),
    }
  );
});

/**
 * POST /api/v1/validate/text
 * Validate text before processing (debugging endpoint)
 */
app.post('/api/v1/validate/text', async (c: Context) => {
  const body = await c.req.json<{ text: string }>();

  if (!body.text) {
    return c.json({ error: 'text is required' }, 400);
  }

  // Import validation utilities
  const { validateText, preprocessText, sentenceTokenize } = await import('./services/tokenizer');

  const validation = validateText(body.text, false);
  const processed = preprocessText(body.text);
  const sentences = sentenceTokenize(processed);

  return c.json({
    valid: validation.valid,
    error: validation.error,
    originalLength: body.text.length,
    processedLength: processed.length,
    sentenceCount: sentences.length,
    preview: processed.slice(0, 500) + (processed.length > 500 ? '...' : ''),
  });
});

// ============================================================
// OCR Endpoints
// ============================================================

/**
 * GET /api/v1/ocr/status
 * Check OCR service availability
 */
app.get('/api/v1/ocr/status', (c) => {
  const providers = getAvailableProviders();
  return c.json({
    available: providers.length > 0,
    providers,
    configured: {
      mistral: isMistralConfigured(),
      hunyuan: isHunyuanAvailable(),
    },
  });
});

/**
 * GET /api/v1/ocr/providers
 * Get detailed provider status
 */
app.get('/api/v1/ocr/providers', (c) => {
  return c.json(getProviderStatus());
});

/**
 * POST /api/v1/ocr
 * Perform OCR on uploaded document
 */
app.post('/api/v1/ocr', async (c: Context) => {
  const providers = getAvailableProviders();

  if (providers.length === 0) {
    return c.json(
      {
        error: 'OCR service not configured',
        hint: 'Set MISTRAL_API_KEY environment variable for cloud OCR',
        available_providers: [],
      },
      503
    );
  }

  try {
    // Parse multipart form or JSON body
    const contentType = c.req.header('Content-Type') || '';

    let documentBuffer: Buffer;
    let options: { provider?: string; format?: string; detailed?: boolean } = {};

    if (contentType.includes('multipart/form-data')) {
      const formData = await c.req.formData();
      const file = formData.get('file') as File;

      if (!file) {
        return c.json({ error: 'No file uploaded' }, 400);
      }

      documentBuffer = Buffer.from(await file.arrayBuffer());
      options.provider = formData.get('provider') as string || undefined;
      options.format = formData.get('format') as string || undefined;
      options.detailed = formData.get('detailed') === 'true';
    } else if (contentType.includes('application/json')) {
      const body = await c.req.json<{
        document: string;  // base64 or URL
        provider?: string;
        format?: string;
        detailed?: boolean;
      }>();

      if (!body.document) {
        return c.json({ error: 'document field is required (base64 or URL)' }, 400);
      }

      // Handle base64 or URL
      if (body.document.startsWith('data:') || body.document.startsWith('http')) {
        // Pass as-is, OCR service will handle it
        const result = await performOCR(body.document, {
          provider: body.provider as any,
          format: body.format as any,
          detailed: body.detailed,
        });

        return c.json({
          text: result.text,
          pages: result.pages,
          provider: result.provider,
          timing: result.timing,
        });
      }

      // Assume base64 encoded
      documentBuffer = Buffer.from(body.document, 'base64');
      options = {
        provider: body.provider,
        format: body.format,
        detailed: body.detailed,
      };
    } else {
      return c.json({ error: 'Unsupported Content-Type. Use multipart/form-data or application/json' }, 400);
    }

    const result = await performOCR(documentBuffer, {
      provider: options.provider as any,
      format: options.format as any,
      detailed: options.detailed,
    });

    return c.json({
      text: result.text,
      pages: result.pages,
      provider: result.provider,
      timing: result.timing,
    }, 200, {
      'X-Processing-Time-Ms': result.timing.total_ms.toFixed(2),
      'X-OCR-Provider': result.provider,
    });
  } catch (error) {
    logger.error('OCR request failed', error);
    return c.json(
      {
        error: error instanceof Error ? error.message : 'OCR processing failed',
        code: 'OCR_ERROR',
      },
      500
    );
  }
});

// ============================================================
// Root / Info
// ============================================================

app.get('/', (c) => {
  const hw = getHardwareStatus();
  const ocrProviders = getAvailableProviders();

  return c.json({
    name: 'Inception ONNX',
    version: '2.0.0',
    provider: hw.provider,
    device: hw.deviceName,
    ocr: {
      available: ocrProviders.length > 0,
      providers: ocrProviders,
    },
    endpoints: [
      'GET  /health               - Health check',
      'GET  /status               - Detailed status',
      'GET  /metrics              - Prometheus metrics',
      'POST /api/v1/embed/query   - Query embedding',
      'POST /api/v1/embed/text    - Document embedding',
      'POST /api/v1/embed/batch   - Batch embeddings',
      'POST /api/v1/validate/text - Text validation',
      'GET  /api/v1/ocr/status    - OCR availability',
      'GET  /api/v1/ocr/providers - OCR provider details',
      'POST /api/v1/ocr           - Perform OCR',
      'POST /api/v1/ocr/batch     - Batch OCR',
    ],
  });
});

// ============================================================
// Server Startup
// ============================================================

const port = parseInt(process.env.PORT || '8005');

// Initialize embedding service on startup
async function startup() {
  logger.info(`Starting Inception ONNX server on port ${port}`);
  logger.info(`Execution provider: ${getHardwareStatus().provider}`);

  try {
    // Pre-initialize the embedding service
    await initEmbeddingService();
    logger.info('Embedding service initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize embedding service on startup', error);
    logger.warn('Service will attempt to initialize on first request');
  }
}

// Run startup
startup().catch(console.error);

export default {
  port,
  fetch: app.fetch,
};

// Export app for testing
export { app };
