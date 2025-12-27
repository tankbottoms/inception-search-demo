/**
 * Configuration management for Inception ONNX service
 * Mirrors Python config from backup/inception/inception/config.py
 */

import type { Settings, ExecutionProvider } from '../types';

function getEnvString(key: string, defaultValue: string): string {
  return process.env[key] || defaultValue;
}

function getEnvInt(key: string, defaultValue: number): number {
  const value = process.env[key];
  if (!value) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
}

function getEnvFloat(key: string, defaultValue: number): number {
  const value = process.env[key];
  if (!value) return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

function getEnvBool(key: string, defaultValue: boolean): boolean {
  const value = process.env[key]?.toLowerCase();
  if (!value) return defaultValue;
  return value === 'true' || value === '1' || value === 'yes';
}

/**
 * Create settings from environment variables
 */
export function createSettings(): Settings {
  const forceCpu = getEnvBool('FORCE_CPU', false);
  const envProvider = getEnvString('EXECUTION_PROVIDER', 'auto');

  let executionProvider: ExecutionProvider;
  if (forceCpu) {
    executionProvider = 'cpu';
  } else if (envProvider === 'cuda' || envProvider === 'gpu') {
    executionProvider = 'cuda';
  } else if (envProvider === 'cpu') {
    executionProvider = 'cpu';
  } else {
    // Auto-detection will happen later in hardware.ts
    executionProvider = 'cpu';
  }

  return {
    // Model settings - matching Python defaults
    modelName: getEnvString(
      'TRANSFORMER_MODEL_NAME',
      'freelawproject/modernbert-embed-base_finetune_512'
    ),
    modelVersion: getEnvString('TRANSFORMER_MODEL_VERSION', 'main'),
    maxTokens: getEnvInt('MAX_TOKENS', 512),
    overlapRatio: getEnvFloat('OVERLAP_RATIO', 0.004),

    // Text constraints
    minTextLength: getEnvInt('MIN_TEXT_LENGTH', 1),
    maxQueryLength: getEnvInt('MAX_QUERY_LENGTH', 1000),
    maxTextLength: getEnvInt('MAX_TEXT_LENGTH', 10_000_000),

    // Processing settings
    maxBatchSize: getEnvInt('MAX_BATCH_SIZE', 100),
    processingBatchSize: getEnvInt('PROCESSING_BATCH_SIZE', 8),
    maxWorkers: getEnvInt('MAX_WORKERS', 4),
    poolTimeout: getEnvInt('POOL_TIMEOUT', 3600),

    // Hardware settings
    forceCpu,
    executionProvider,

    // Feature flags
    enableMetrics: getEnvBool('ENABLE_METRICS', true),
  };
}

// Singleton settings instance
export const settings = createSettings();

// Model cache directory
export const MODEL_CACHE_DIR = getEnvString('MODEL_CACHE_DIR', './models');

// Prefixes for embeddings (from registry.json)
export const QUERY_PREFIX = 'search_query: ';
export const DOCUMENT_PREFIX = 'search_document: ';
