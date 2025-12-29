/**
 * PDF Utilities - Extract text from PDFs
 *
 * Uses pdf-parse for text-based PDFs.
 * For scanned PDFs without embedded text, falls back to OCR on images.
 */

import { PDFParse } from 'pdf-parse';
import { readFileSync, existsSync } from 'fs';
import { logger, Timer } from '../logger';

export interface PDFPage {
  page: number;
  text: string;
}

export interface PDFExtractResult {
  text: string;
  pages: PDFPage[];
  pageCount: number;
  hasText: boolean;
  processingTime: number;
  metadata?: {
    title?: string;
    author?: string;
    subject?: string;
    creator?: string;
  };
}

/**
 * Check if a buffer or data URL is a PDF
 */
export function isPDF(input: Buffer | string): boolean {
  if (Buffer.isBuffer(input)) {
    // Check PDF magic bytes: %PDF-
    return input.length >= 5 && input.slice(0, 5).toString() === '%PDF-';
  }

  if (typeof input === 'string') {
    // Check for base64 data URL with PDF MIME type
    if (input.startsWith('data:application/pdf')) {
      return true;
    }

    // Check if it's a file path ending in .pdf
    if (input.endsWith('.pdf') && existsSync(input)) {
      return true;
    }

    // Try to decode base64 and check magic bytes
    try {
      const buffer = Buffer.from(input.replace(/^data:[^;]+;base64,/, ''), 'base64');
      return buffer.length >= 5 && buffer.slice(0, 5).toString() === '%PDF-';
    } catch {
      return false;
    }
  }

  return false;
}

/**
 * Convert input to a Buffer
 */
export function toBuffer(input: Buffer | string): Buffer {
  if (Buffer.isBuffer(input)) {
    return input;
  }

  if (typeof input === 'string') {
    // File path
    if (existsSync(input)) {
      return readFileSync(input);
    }

    // Base64 data URL
    if (input.includes(';base64,')) {
      const base64Data = input.split(';base64,')[1];
      return Buffer.from(base64Data, 'base64');
    }

    // Raw base64
    return Buffer.from(input, 'base64');
  }

  throw new Error('Invalid input: expected Buffer or string');
}

/**
 * Extract text from a PDF using pdf-parse
 */
export async function extractTextFromPDF(
  input: Buffer | string
): Promise<PDFExtractResult> {
  const timer = new Timer('pdf-extract');

  try {
    const buffer = toBuffer(input);

    if (!isPDF(buffer)) {
      throw new Error('Input is not a valid PDF');
    }

    timer.checkpoint('loaded');

    // Create PDFParse instance with buffer data (converts Buffer to Uint8Array automatically)
    const parser = new PDFParse({ data: buffer });

    // Get text from PDF (loading happens automatically)
    const textResult = await parser.getText();

    timer.checkpoint('parsed');

    // Get info
    const infoResult = await parser.getInfo();
    const pageCount = textResult.pages?.length || 1;

    // Combine text from all pages
    const allText = textResult.pages?.map((p) => p.text).join('\n\n') || textResult.text || '';

    // Build pages array
    const pages: PDFPage[] = textResult.pages?.map((p, index: number) => ({
      page: index + 1,
      text: (p.text || '').trim(),
    })) || [];

    // If no pages, treat entire text as one page
    if (pages.length === 0 && allText.trim()) {
      pages.push({ page: 1, text: allText.trim() });
    }

    // Clean up
    await parser.destroy();

    // Extract metadata from info dictionary (stored in infoResult.info)
    const pdfInfo = infoResult.info as Record<string, unknown> | undefined;

    const result: PDFExtractResult = {
      text: allText.trim(),
      pages,
      pageCount,
      hasText: allText.trim().length > 0,
      processingTime: timer.elapsed(),
      metadata: pdfInfo ? {
        title: pdfInfo.Title as string | undefined,
        author: pdfInfo.Author as string | undefined,
        subject: pdfInfo.Subject as string | undefined,
        creator: pdfInfo.Creator as string | undefined,
      } : undefined,
    };

    timer.log(`PDF extracted (${result.pageCount} pages, ${result.text.length} chars, hasText: ${result.hasText})`);

    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    logger.error(`PDF extraction failed: ${errorMessage}`);
    throw new Error(`PDF extraction failed: ${errorMessage}`);
  }
}

/**
 * Check if a PDF has extractable text (not scanned)
 */
export async function pdfHasText(input: Buffer | string): Promise<boolean> {
  try {
    const result = await extractTextFromPDF(input);
    // Consider it as having text if there's at least 50 characters of meaningful content
    return result.hasText && result.text.length >= 50;
  } catch {
    return false;
  }
}

export default {
  isPDF,
  toBuffer,
  extractTextFromPDF,
  pdfHasText,
};
