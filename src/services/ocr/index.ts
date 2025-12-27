/**
 * OCR Services - Unified OCR interface
 *
 * Supports multiple OCR backends:
 * - Mistral Pixtral (cloud API)
 * - HunyuanOCR (local ONNX inference - CPU/CUDA)
 */

import {
  performOCR as mistralOCR,
  performBatchOCR as mistralBatchOCR,
  isMistralConfigured,
  type OCRResult,
  type OCROptions,
} from './mistral';
import {
  performOCR as hunyuanOCR,
  performBatchOCR as hunyuanBatchOCR,
  isHunyuanAvailable,
  initializeHunyuanOCR,
  getProviderInfo as getHunyuanProviderInfo,
  shutdown as shutdownHunyuan,
} from './hunyuan';
import { logger } from '../logger';

export type OCRProvider = 'mistral' | 'hunyuan' | 'auto';

export interface UnifiedOCROptions extends OCROptions {
  provider?: OCRProvider;
  maxTokens?: number;
  detailed?: boolean;
}

/**
 * Get available OCR providers
 */
export function getAvailableProviders(): OCRProvider[] {
  const available: OCRProvider[] = [];

  // Check for local HunyuanOCR (preferred)
  if (isHunyuanAvailable()) {
    available.push('hunyuan');
  }

  // Check for Mistral cloud API
  if (isMistralConfigured()) {
    available.push('mistral');
  }

  return available;
}

/**
 * Get provider status with details
 */
export function getProviderStatus(): Record<OCRProvider, { available: boolean; details?: string }> {
  const hunyuanInfo = getHunyuanProviderInfo();

  return {
    hunyuan: {
      available: isHunyuanAvailable(),
      details: hunyuanInfo.initialized
        ? `Initialized with ${hunyuanInfo.provider.toUpperCase()} provider`
        : 'Not initialized',
    },
    mistral: {
      available: isMistralConfigured(),
      details: isMistralConfigured() ? 'API key configured' : 'MISTRAL_API_KEY not set',
    },
    auto: {
      available: getAvailableProviders().length > 0,
      details: `Will use: ${getAvailableProviders()[0] || 'none'}`,
    },
  };
}

/**
 * Select the best available OCR provider
 */
function selectProvider(requested?: OCRProvider): OCRProvider {
  const available = getAvailableProviders();

  if (available.length === 0) {
    throw new Error(
      'No OCR providers available. ' +
      'Install HunyuanOCR model or configure MISTRAL_API_KEY for Mistral OCR.'
    );
  }

  if (requested && requested !== 'auto') {
    if (!available.includes(requested)) {
      throw new Error(`OCR provider '${requested}' is not available. Available: ${available.join(', ')}`);
    }
    return requested;
  }

  // Auto-select: prefer local providers (hunyuan), fallback to cloud (mistral)
  if (available.includes('hunyuan')) return 'hunyuan';
  if (available.includes('mistral')) return 'mistral';

  throw new Error('No OCR provider could be selected');
}

/**
 * Perform OCR with automatic provider selection
 */
export async function performOCR(
  document: Buffer | string,
  options: UnifiedOCROptions = {}
): Promise<OCRResult & { provider: OCRProvider }> {
  const provider = selectProvider(options.provider);

  logger.info(`Performing OCR with provider: ${provider}`);

  switch (provider) {
    case 'mistral': {
      const result = await mistralOCR(document, options);
      return { ...result, provider };
    }

    case 'hunyuan': {
      // Initialize if needed
      await initializeHunyuanOCR();

      const result = await hunyuanOCR(document, {
        maxTokens: options.maxTokens,
        detailed: options.detailed,
      });

      // Convert HunyuanOCRResult to OCRResult format
      return {
        text: result.text,
        pages: [{
          page: 1,
          text: result.text,
          confidence: result.confidence,
        }],
        timing: {
          total_ms: result.processingTime,
        },
        provider,
      };
    }

    default:
      throw new Error(`Unknown OCR provider: ${provider}`);
  }
}

/**
 * Perform batch OCR with automatic provider selection
 */
export async function performBatchOCR(
  documents: Array<Buffer | string>,
  options: UnifiedOCROptions = {}
): Promise<OCRResult & { provider: OCRProvider }> {
  const provider = selectProvider(options.provider);

  logger.info(`Performing batch OCR with provider: ${provider}, documents: ${documents.length}`);

  switch (provider) {
    case 'mistral': {
      const result = await mistralBatchOCR(documents, options);
      return { ...result, provider };
    }

    case 'hunyuan': {
      // Initialize if needed
      await initializeHunyuanOCR();

      const results = await hunyuanBatchOCR(documents, {
        maxTokens: options.maxTokens,
        detailed: options.detailed,
      });

      // Combine results
      const combinedText = results.map(r => r.text).join('\n\n---\n\n');
      const totalTime = results.reduce((sum, r) => sum + r.processingTime, 0);
      const pages = results.map((r, i) => ({
        page: i + 1,
        text: r.text,
        confidence: r.confidence,
      }));

      return {
        text: combinedText,
        pages,
        timing: {
          total_ms: totalTime,
        },
        provider,
      };
    }

    default:
      throw new Error(`Unknown OCR provider: ${provider}`);
  }
}

/**
 * Initialize OCR providers (pre-warm for faster first request)
 */
export async function initializeOCR(provider?: OCRProvider): Promise<void> {
  const targetProvider = provider || selectProvider();

  if (targetProvider === 'hunyuan') {
    await initializeHunyuanOCR();
    logger.info('HunyuanOCR initialized');
  }
  // Mistral doesn't need initialization
}

/**
 * Shutdown OCR providers and release resources
 */
export async function shutdownOCR(): Promise<void> {
  await shutdownHunyuan();
  logger.info('OCR providers shutdown complete');
}

// Re-export types and individual providers
export type { OCRResult, OCROptions } from './mistral';
export { isMistralConfigured } from './mistral';
export { isHunyuanAvailable, getProviderInfo as getHunyuanInfo } from './hunyuan';
