/**
 * PDF Text Extraction and Comparison Utilities
 * Uses pdf-parse for extracting embedded text from PDFs
 */

import fs from 'fs';
import { PDFParse } from 'pdf-parse';

export interface PdfTextResult {
    text: string;
    charCount: number;
    pageCount: number;
    success: boolean;
    error?: string;
}

export interface TextComparisonResult {
    rawTextChars: number;
    ocrTextChars: number;
    similarityPercent: number;
    differencePercent: number;
    rawTextAvailable: boolean;
}

/**
 * Extract embedded text from a PDF file using pdf-parse
 * This extracts selectable text only - scanned documents will return empty/minimal text
 */
export async function extractPdfText(filePath: string): Promise<PdfTextResult> {
    try {
        const buffer = await fs.promises.readFile(filePath);
        const parser = new PDFParse({ data: buffer });
        await parser.load();
        const textResult = await parser.getText();

        // Combine text from all pages
        let fullText = '';
        let pageCount = 0;

        if (textResult.pages) {
            pageCount = textResult.pages.length;
            for (const page of textResult.pages) {
                // Each page has a 'text' property with the page content
                if (page.text) {
                    fullText += page.text + '\n\n';
                }
            }
        }

        fullText = fullText.trim();

        // Clean up
        await parser.destroy();

        return {
            text: fullText,
            charCount: fullText.length,
            pageCount: pageCount,
            success: true
        };
    } catch (error) {
        return {
            text: '',
            charCount: 0,
            pageCount: 0,
            success: false,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}

/**
 * Calculate text similarity using Jaccard similarity on word sets
 * Returns percentage similarity between two text strings
 */
export function calculateTextSimilarity(text1: string, text2: string): number {
    if (!text1 || !text2) return 0;
    if (text1.length === 0 && text2.length === 0) return 100;
    if (text1.length === 0 || text2.length === 0) return 0;

    // Normalize texts: lowercase, remove extra whitespace
    const normalize = (text: string) =>
        text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

    const normalized1 = normalize(text1);
    const normalized2 = normalize(text2);

    // Create word sets
    const words1 = new Set(normalized1.split(' ').filter(w => w.length > 2));
    const words2 = new Set(normalized2.split(' ').filter(w => w.length > 2));

    if (words1.size === 0 && words2.size === 0) return 100;
    if (words1.size === 0 || words2.size === 0) return 0;

    // Calculate Jaccard similarity: intersection / union
    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    const similarity = (intersection.size / union.size) * 100;
    return Math.round(similarity * 10) / 10; // Round to 1 decimal
}

/**
 * Compare raw PDF text extraction with OCR output
 */
export function compareTextResults(rawText: string, ocrText: string): TextComparisonResult {
    const rawChars = rawText.length;
    const ocrChars = ocrText.length;
    const hasRawText = rawChars > 100; // Minimum threshold for meaningful text

    const similarityPercent = hasRawText
        ? calculateTextSimilarity(rawText, ocrText)
        : 0;

    const differencePercent = hasRawText
        ? Math.round((100 - similarityPercent) * 10) / 10
        : 100;

    return {
        rawTextChars: rawChars,
        ocrTextChars: ocrChars,
        similarityPercent,
        differencePercent,
        rawTextAvailable: hasRawText
    };
}

/**
 * Get page count from PDF without full text extraction
 */
export async function getPdfPageCount(filePath: string): Promise<number> {
    try {
        const buffer = await fs.promises.readFile(filePath);
        const parser = new PDFParse({ data: buffer });
        await parser.load();
        const info = await parser.getInfo();
        await parser.destroy();
        return info.numPages || 0;
    } catch {
        return 0;
    }
}
