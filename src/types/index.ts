/**
 * Type definitions for Inception ONNX inference service
 * Mirrors the Python schemas from backup/inception/inception/schemas.py
 */

// ============================================================
// Request Types
// ============================================================

export interface TextRequest {
  id: number;
  text: string;
}

export interface BatchTextRequest {
  documents: TextRequest[];
}

export interface QueryRequest {
  text: string;
}

// ============================================================
// Response Types
// ============================================================

export interface ChunkEmbedding {
  chunk_number: number;
  chunk: string;
  embedding: number[];
}

export interface TextResponse {
  id: number | null;
  embeddings: ChunkEmbedding[];
}

export interface QueryResponse {
  embedding: number[];
}

export interface BatchResponse {
  results: TextResponse[];
  timing: TimingInfo;
}

// ============================================================
// Model Registry Types
// ============================================================

export interface ModelConfig {
  max_tokens?: number;
  embedding_dim?: number;
  pooling?: 'mean' | 'cls' | 'max';
  normalize?: boolean;
  query_prefix?: string;
  document_prefix?: string;
  max_image_size?: number;
}

export interface ModelEntry {
  id: string;
  name: string;
  type: 'embedding' | 'ocr' | 'inference';
  enabled: boolean;
  status?: 'planned' | 'available' | 'loaded';
  config?: ModelConfig;
}

export interface ModelRegistry {
  version: string;
  cache_dir: string;
  models: ModelEntry[];
}

// ============================================================
// Hardware / Provider Types
// ============================================================

export type ExecutionProvider = 'cuda' | 'cpu';

export interface HardwareInfo {
  provider: ExecutionProvider;
  deviceName: string;
  memoryTotal?: number;
  memoryFree?: number;
  cudaVersion?: string;
  computeCapability?: string;
}

// ============================================================
// Timing / Metrics Types
// ============================================================

export interface TimingInfo {
  total_ms: number;
  chunking_ms?: number;
  inference_ms?: number;
  postprocess_ms?: number;
}

export interface OperationMetrics {
  operation: string;
  startTime: number;
  endTime?: number;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
}

// ============================================================
// Configuration Types
// ============================================================

export interface Settings {
  // Model settings
  modelName: string;
  modelVersion: string;
  maxTokens: number;
  overlapRatio: number;

  // Text constraints
  minTextLength: number;
  maxQueryLength: number;
  maxTextLength: number;

  // Processing settings
  maxBatchSize: number;
  processingBatchSize: number;
  maxWorkers: number;
  poolTimeout: number;

  // Hardware settings
  forceCpu: boolean;
  executionProvider: ExecutionProvider;

  // Feature flags
  enableMetrics: boolean;
}

// ============================================================
// Error Types
// ============================================================

export class InferenceError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = 'InferenceError';
  }
}

export class ModelNotFoundError extends InferenceError {
  constructor(modelId: string) {
    super(`Model not found: ${modelId}`, 'MODEL_NOT_FOUND', 404);
    this.name = 'ModelNotFoundError';
  }
}

export class ValidationError extends InferenceError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR', 400);
    this.name = 'ValidationError';
  }
}

// ============================================================
// ONNX Runtime Types
// ============================================================

export interface OnnxSessionOptions {
  executionProviders: string[];
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  logSeverityLevel?: number;
  logVerbosityLevel?: number;
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
}

export interface TokenizerOutput {
  input_ids: bigint[] | number[];
  attention_mask: bigint[] | number[];
  token_type_ids?: bigint[] | number[];
}

export interface ModelOutput {
  last_hidden_state: Float32Array;
  shape: number[];
}
