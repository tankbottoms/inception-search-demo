/**
 * Inception ONNX - TypeScript/Bun Inference Backend
 *
 * Multi-platform ONNX inference service with ARM64 CPU and CUDA GPU acceleration
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';

const app = new Hono();

// Middleware
app.use('*', cors());
app.use('*', logger());

// Health check
app.get('/health', (c) => {
  return c.json({
    status: 'ok',
    version: '2.0.0',
    provider: process.env.EXECUTION_PROVIDER || 'cpu',
    timestamp: new Date().toISOString()
  });
});

// Metrics placeholder
app.get('/metrics', (c) => {
  return c.text('# TODO: Prometheus metrics\n');
});

// API routes placeholder
app.post('/api/v1/embed/query', async (c) => {
  return c.json({ error: 'Not yet implemented' }, 501);
});

app.post('/api/v1/embed/text', async (c) => {
  return c.json({ error: 'Not yet implemented' }, 501);
});

app.post('/api/v1/embed/batch', async (c) => {
  return c.json({ error: 'Not yet implemented' }, 501);
});

app.post('/api/v1/ocr', async (c) => {
  return c.json({ error: 'Not yet implemented' }, 501);
});

// Root
app.get('/', (c) => {
  return c.json({
    name: 'Inception ONNX',
    version: '2.0.0',
    endpoints: [
      'GET /health',
      'GET /metrics',
      'POST /api/v1/embed/query',
      'POST /api/v1/embed/text',
      'POST /api/v1/embed/batch',
      'POST /api/v1/ocr'
    ]
  });
});

const port = parseInt(process.env.PORT || '8005');

console.log(`[INFO] Starting Inception ONNX server on port ${port}`);
console.log(`[INFO] Execution provider: ${process.env.EXECUTION_PROVIDER || 'cpu'}`);

export default {
  port,
  fetch: app.fetch,
};
