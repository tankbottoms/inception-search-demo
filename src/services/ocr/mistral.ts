/**
 * Mistral OCR Client
 *
 * Provides OCR capabilities via Mistral's API
 */

import { logger, Timer } from '../logger';
import { settings } from '../config';

const MISTRAL_API_URL = 'https://api.mistral.ai/v1/chat/completions';
const MISTRAL_OCR_MODEL = 'pixtral-large-latest';

export interface OCRResult {
  text: string;
  pages: PageResult[];
  timing: {
    total_ms: number;
    api_ms?: number;
  };
}

export interface PageResult {
  page: number;
  text: string;
  confidence?: number;
}

export interface OCROptions {
  detailed?: boolean;
  language?: string;
  format?: 'text' | 'markdown' | 'json';
}

/**
 * Check if Mistral API is configured
 */
export function isMistralConfigured(): boolean {
  return !!process.env.MISTRAL_API_KEY;
}

/**
 * Get Mistral API key from environment
 */
function getApiKey(): string {
  const key = process.env.MISTRAL_API_KEY;
  if (!key) {
    throw new Error('MISTRAL_API_KEY environment variable not set');
  }
  return key;
}

/**
 * Encode image/PDF to base64 data URI
 */
function encodeToDataUri(buffer: Buffer, mimeType: string): string {
  const base64 = buffer.toString('base64');
  return `data:${mimeType};base64,${base64}`;
}

/**
 * Detect MIME type from buffer
 */
function detectMimeType(buffer: Buffer): string {
  // Check magic bytes
  if (buffer[0] === 0x25 && buffer[1] === 0x50 && buffer[2] === 0x44 && buffer[3] === 0x46) {
    return 'application/pdf';
  }
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return 'image/png';
  }
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return 'image/jpeg';
  }
  if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) {
    return 'image/gif';
  }
  if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) {
    return 'image/webp';
  }
  // Default to octet-stream
  return 'application/octet-stream';
}

/**
 * Build OCR prompt based on options
 */
function buildPrompt(options: OCROptions): string {
  const format = options.format || 'markdown';
  const detailed = options.detailed ?? false;

  let prompt = 'Extract all text content from this document.';

  if (detailed) {
    prompt += ' Preserve the document structure including headings, paragraphs, lists, and tables.';
  }

  switch (format) {
    case 'json':
      prompt += ' Return the result as valid JSON with keys: title, sections (array of {heading, content}), tables (array of {headers, rows}).';
      break;
    case 'markdown':
      prompt += ' Format the output as clean Markdown with proper headings and formatting.';
      break;
    case 'text':
      prompt += ' Return plain text without formatting.';
      break;
  }

  if (options.language) {
    prompt += ` The document is in ${options.language}.`;
  }

  return prompt;
}

/**
 * Process OCR response from Mistral
 */
function parseOCRResponse(response: any, options: OCROptions): OCRResult['pages'] {
  const content = response.choices?.[0]?.message?.content || '';

  // For single document, return as single page
  return [{
    page: 1,
    text: content,
    confidence: 1.0,
  }];
}

/**
 * Perform OCR on a document using Mistral API
 */
export async function performOCR(
  document: Buffer | string,
  options: OCROptions = {}
): Promise<OCRResult> {
  const timer = new Timer('ocr-mistral');

  if (!isMistralConfigured()) {
    throw new Error('Mistral API not configured. Set MISTRAL_API_KEY environment variable.');
  }

  const apiKey = getApiKey();

  // Handle document input
  let dataUri: string;
  let mimeType: string;

  if (typeof document === 'string') {
    // Assume it's already a data URI or URL
    if (document.startsWith('data:')) {
      dataUri = document;
    } else {
      // It's a URL, fetch it
      const response = await fetch(document);
      const buffer = Buffer.from(await response.arrayBuffer());
      mimeType = detectMimeType(buffer);
      dataUri = encodeToDataUri(buffer, mimeType);
    }
  } else {
    mimeType = detectMimeType(document);
    dataUri = encodeToDataUri(document, mimeType);
  }

  timer.checkpoint('prepared');

  // Build request
  const prompt = buildPrompt(options);

  const requestBody = {
    model: MISTRAL_OCR_MODEL,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: prompt,
          },
          {
            type: 'image_url',
            image_url: dataUri,
          },
        ],
      },
    ],
    max_tokens: 16384,
  };

  logger.debug('Calling Mistral OCR API', { model: MISTRAL_OCR_MODEL });

  const apiStart = Date.now();
  const response = await fetch(MISTRAL_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    logger.error('Mistral API error', { status: response.status, error });
    throw new Error(`Mistral API error: ${response.status} - ${error}`);
  }

  const result = await response.json();
  const apiMs = Date.now() - apiStart;
  timer.checkpoint('api-complete');

  const pages = parseOCRResponse(result, options);
  const fullText = pages.map(p => p.text).join('\n\n');

  timer.log('OCR complete');

  return {
    text: fullText,
    pages,
    timing: {
      total_ms: timer.elapsed(),
      api_ms: apiMs,
    },
  };
}

/**
 * Perform OCR on multiple pages/images
 */
export async function performBatchOCR(
  documents: Array<Buffer | string>,
  options: OCROptions = {}
): Promise<OCRResult> {
  const timer = new Timer('ocr-batch');
  const pages: PageResult[] = [];
  let totalApiMs = 0;

  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    const result = await performOCR(doc, options);

    for (const page of result.pages) {
      pages.push({
        ...page,
        page: i + 1,
      });
    }

    if (result.timing.api_ms) {
      totalApiMs += result.timing.api_ms;
    }
  }

  const fullText = pages.map(p => p.text).join('\n\n');

  timer.log('Batch OCR complete');

  return {
    text: fullText,
    pages,
    timing: {
      total_ms: timer.elapsed(),
      api_ms: totalApiMs,
    },
  };
}

// Export OCR service
export default {
  isMistralConfigured,
  performOCR,
  performBatchOCR,
};
