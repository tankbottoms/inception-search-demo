/**
 * Tesseract OCR - CPU-compatible local OCR
 *
 * Uses Tesseract.js for OCR that works on all platforms (M1, x86, ARM64).
 * No GPU required - pure JavaScript/WebAssembly implementation.
 */

import Tesseract from 'tesseract.js';
import { existsSync, readFileSync } from 'fs';
import { logger, Timer } from '../logger';

export interface TesseractOCRResult {
  text: string;
  confidence: number;
  processingTime: number;
  provider: 'tesseract';
  words?: Array<{
    text: string;
    confidence: number;
    bbox: { x0: number; y0: number; x1: number; y1: number };
  }>;
}

export interface TesseractOCROptions {
  language?: string;
  detailed?: boolean;
}

// Cached worker for reuse
let worker: Tesseract.Worker | null = null;
let isInitialized = false;
let currentLanguage = 'eng';

/**
 * Check if Tesseract is available
 */
export function isTesseractAvailable(): boolean {
  try {
    // Tesseract.js is always available if the package is installed
    return typeof Tesseract !== 'undefined';
  } catch {
    return false;
  }
}

/**
 * Initialize Tesseract worker
 */
export async function initializeTesseract(language: string = 'eng'): Promise<void> {
  if (isInitialized && currentLanguage === language) {
    return;
  }

  const timer = new Timer('tesseract-init');
  logger.info(`Initializing Tesseract OCR (language: ${language})...`);

  // Terminate existing worker if language changed
  if (worker && currentLanguage !== language) {
    await worker.terminate();
    worker = null;
  }

  if (!worker) {
    worker = await Tesseract.createWorker(language, 1, {
      logger: (m) => {
        if (m.status === 'recognizing text') {
          // Progress logging can be verbose, skip or log sparingly
        }
      },
    });
  }

  currentLanguage = language;
  isInitialized = true;
  timer.log('Tesseract initialized');
}

/**
 * Perform OCR on an image
 */
export async function performOCR(
  image: Buffer | string,
  options: TesseractOCROptions = {}
): Promise<TesseractOCRResult> {
  const timer = new Timer('tesseract-ocr');
  const { language = 'eng', detailed = false } = options;

  // Initialize if needed
  if (!isInitialized || currentLanguage !== language) {
    await initializeTesseract(language);
  }

  if (!worker) {
    throw new Error('Tesseract worker not initialized');
  }

  // Handle input
  let imageData: Buffer | string;
  if (typeof image === 'string') {
    if (existsSync(image)) {
      imageData = readFileSync(image);
    } else if (image.startsWith('data:image')) {
      // Base64 data URL
      const base64Data = image.split(',')[1];
      imageData = Buffer.from(base64Data, 'base64');
    } else {
      // Assume raw base64
      imageData = Buffer.from(image, 'base64');
    }
  } else {
    imageData = image;
  }

  timer.checkpoint('prepared');

  // Perform recognition
  const result = await worker.recognize(imageData);
  timer.checkpoint('recognized');

  const processingTime = timer.elapsed();

  // Build result
  const ocrResult: TesseractOCRResult = {
    text: result.data.text.trim(),
    confidence: result.data.confidence / 100, // Normalize to 0-1
    processingTime,
    provider: 'tesseract',
  };

  // Add word-level details if requested
  if (detailed && result.data.words) {
    ocrResult.words = result.data.words.map((word) => ({
      text: word.text,
      confidence: word.confidence / 100,
      bbox: word.bbox,
    }));
  }

  timer.log(`OCR complete (${result.data.text.length} chars, ${(result.data.confidence).toFixed(1)}% confidence)`);

  return ocrResult;
}

/**
 * Perform batch OCR
 */
export async function performBatchOCR(
  images: Array<Buffer | string>,
  options: TesseractOCROptions = {}
): Promise<TesseractOCRResult[]> {
  const results: TesseractOCRResult[] = [];

  for (const image of images) {
    const result = await performOCR(image, options);
    results.push(result);
  }

  return results;
}

/**
 * Get provider info
 */
export function getProviderInfo(): { provider: 'tesseract'; initialized: boolean; language: string } {
  return {
    provider: 'tesseract',
    initialized: isInitialized,
    language: currentLanguage,
  };
}

/**
 * Shutdown and release resources
 */
export async function shutdown(): Promise<void> {
  if (worker) {
    await worker.terminate();
    worker = null;
  }
  isInitialized = false;
  logger.info('Tesseract OCR shutdown complete');
}

export default {
  isTesseractAvailable,
  initializeTesseract,
  performOCR,
  performBatchOCR,
  getProviderInfo,
  shutdown,
};
