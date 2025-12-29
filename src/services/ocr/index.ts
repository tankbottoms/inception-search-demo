/**
 * OCR Services - Unified OCR interface
 *
 * Supports multiple OCR backends:
 * - Tesseract (local CPU - works on all platforms including M1)
 * - Mistral Pixtral (cloud API)
 * - HunyuanOCR (local ONNX inference - requires GPU for best performance)
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
import {
  performOCR as tesseractOCR,
  performBatchOCR as tesseractBatchOCR,
  isTesseractAvailable,
  initializeTesseract,
  getProviderInfo as getTesseractProviderInfo,
  shutdown as shutdownTesseract,
} from './tesseract';
import {
  isPDF,
  extractTextFromPDF,
  pdfHasText,
} from './pdf-utils';
import { logger, Timer } from '../logger';

export type OCRProvider = 'tesseract' | 'mistral' | 'hunyuan' | 'auto';

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

  // Check for Tesseract (always available, works on CPU/M1)
  if (isTesseractAvailable()) {
    available.push('tesseract');
  }

  // Check for local HunyuanOCR (requires ONNX model)
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
  const tesseractInfo = getTesseractProviderInfo();

  return {
    tesseract: {
      available: isTesseractAvailable(),
      details: tesseractInfo.initialized
        ? `Initialized (language: ${tesseractInfo.language})`
        : 'Available (CPU-based, works on all platforms)',
    },
    hunyuan: {
      available: isHunyuanAvailable(),
      details: hunyuanInfo.initialized
        ? `Initialized with ${hunyuanInfo.provider.toUpperCase()} provider`
        : 'ONNX model not found',
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
 * Priority: tesseract (always works) > hunyuan (GPU) > mistral (cloud)
 */
function selectProvider(requested?: OCRProvider): OCRProvider {
  const available = getAvailableProviders();

  if (available.length === 0) {
    throw new Error(
      'No OCR providers available. ' +
      'Install tesseract.js, HunyuanOCR model, or configure MISTRAL_API_KEY.'
    );
  }

  if (requested && requested !== 'auto') {
    if (!available.includes(requested)) {
      throw new Error(`OCR provider '${requested}' is not available. Available: ${available.join(', ')}`);
    }
    return requested;
  }

  // Auto-select priority: tesseract (CPU, always works) > hunyuan (GPU) > mistral (cloud)
  if (available.includes('tesseract')) return 'tesseract';
  if (available.includes('hunyuan')) return 'hunyuan';
  if (available.includes('mistral')) return 'mistral';

  throw new Error('No OCR provider could be selected');
}

/**
 * Perform OCR with automatic provider selection
 * Handles PDFs by extracting text directly when possible
 */
export async function performOCR(
  document: Buffer | string,
  options: UnifiedOCROptions = {}
): Promise<OCRResult & { provider: OCRProvider }> {
  const timer = new Timer('ocr-unified');

  // Check if document is a PDF
  if (isPDF(document)) {
    logger.info('Detected PDF document, attempting text extraction...');

    try {
      // Try to extract text from PDF directly (works for text-based PDFs)
      const pdfResult = await extractTextFromPDF(document);

      if (pdfResult.hasText && pdfResult.text.length >= 50) {
        logger.info(`PDF text extraction successful: ${pdfResult.pageCount} pages, ${pdfResult.text.length} chars`);
        return {
          text: pdfResult.text,
          pages: pdfResult.pages.map(p => ({
            page: p.page,
            text: p.text,
            confidence: 1.0, // Direct text extraction has perfect confidence
          })),
          timing: {
            total_ms: pdfResult.processingTime,
          },
          provider: 'tesseract' as OCRProvider, // Use tesseract as the nominal provider for PDF extraction
        };
      }

      // PDF is likely scanned (no embedded text)
      logger.info('PDF appears to be scanned (no embedded text), falling back to OCR provider...');

      // For scanned PDFs, try cloud providers first as they handle PDFs better
      const available = getAvailableProviders();
      if (available.includes('mistral') && (!options.provider || options.provider === 'auto')) {
        logger.info('Using Mistral for scanned PDF OCR');
        const result = await mistralOCR(document, options);
        return { ...result, provider: 'mistral' };
      }

      if (available.includes('hunyuan') && (!options.provider || options.provider === 'auto')) {
        logger.info('Using HunyuanOCR for scanned PDF OCR');
        await initializeHunyuanOCR();
        const result = await hunyuanOCR(document, {
          maxTokens: options.maxTokens,
          detailed: options.detailed,
        });
        return {
          text: result.text,
          pages: [{ page: 1, text: result.text, confidence: result.confidence }],
          timing: { total_ms: result.processingTime },
          provider: 'hunyuan',
        };
      }

      // No cloud providers available, return what we have from PDF extraction
      throw new Error(
        'Scanned PDF detected but no OCR provider available for image-based OCR. ' +
        'Configure MISTRAL_API_KEY for cloud OCR, or provide image files instead of PDFs for Tesseract.'
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      // If it's our known error about scanned PDFs, re-throw it
      if (errorMessage.includes('Scanned PDF detected')) {
        throw error;
      }

      logger.warn(`PDF extraction failed: ${errorMessage}, trying OCR providers...`);
      // Fall through to regular OCR if PDF extraction fails
    }
  }

  // Regular OCR for images
  const provider = selectProvider(options.provider);

  logger.info(`Performing OCR with provider: ${provider}`);

  switch (provider) {
    case 'tesseract': {
      // Initialize if needed
      await initializeTesseract();

      const result = await tesseractOCR(document, {
        detailed: options.detailed,
      });

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
    case 'tesseract': {
      await initializeTesseract();

      const results = await tesseractBatchOCR(documents, {
        detailed: options.detailed,
      });

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

    case 'mistral': {
      const result = await mistralBatchOCR(documents, options);
      return { ...result, provider };
    }

    case 'hunyuan': {
      await initializeHunyuanOCR();

      const results = await hunyuanBatchOCR(documents, {
        maxTokens: options.maxTokens,
        detailed: options.detailed,
      });

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

  switch (targetProvider) {
    case 'tesseract':
      await initializeTesseract();
      logger.info('Tesseract OCR initialized');
      break;
    case 'hunyuan':
      await initializeHunyuanOCR();
      logger.info('HunyuanOCR initialized');
      break;
    case 'mistral':
      // Mistral doesn't need initialization
      logger.info('Mistral OCR ready (cloud API)');
      break;
  }
}

/**
 * Shutdown OCR providers and release resources
 */
export async function shutdownOCR(): Promise<void> {
  await Promise.all([
    shutdownTesseract(),
    shutdownHunyuan(),
  ]);
  logger.info('OCR providers shutdown complete');
}

// Re-export types and individual providers
export type { OCRResult, OCROptions } from './mistral';
export { isMistralConfigured } from './mistral';
export { isHunyuanAvailable, getProviderInfo as getHunyuanInfo } from './hunyuan';
export { isTesseractAvailable, getProviderInfo as getTesseractInfo } from './tesseract';
