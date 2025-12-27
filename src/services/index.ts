/**
 * Services barrel export
 */

export { settings, MODEL_CACHE_DIR, QUERY_PREFIX, DOCUMENT_PREFIX } from './config';
export { logger, Timer } from './logger';
export { detectHardware, getExecutionProviders, getCurrentProvider, getHardwareStatus } from './hardware';
export { loadModel, loadRegistry, getModelEntry, checkModelAvailable, unloadModel, getLoadedModels } from './model-loader';
export type { LoadedModel, TokenizerConfig, ModelFiles } from './model-loader';
export { SimpleTokenizer, TextChunker, sentenceTokenize, preprocessText, validateText } from './tokenizer';
export {
  initEmbeddingService,
  generateQueryEmbedding,
  generateTextEmbedding,
  generateBatchEmbeddings,
  getServiceStatus,
  shutdownEmbeddingService,
} from './embedding';
export { metrics, registry } from './metrics';

// OCR Services
export {
  performOCR,
  performBatchOCR,
  getAvailableProviders,
  getProviderStatus,
  initializeOCR,
  shutdownOCR,
  isMistralConfigured,
  isHunyuanAvailable,
} from './ocr';
export type { OCRResult, OCROptions, OCRProvider, UnifiedOCROptions } from './ocr';
