import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import axios from 'axios';
import { extractPdfText, compareTextResults, type TextComparisonResult } from './pdf-utils';

const INCEPTION_API = process.env.INCEPTION_URL || 'http://localhost:8005';
const OCR_DIR = 'ocr';

export interface ExtractTextResult {
    text: string;
    duration: number;
    rawText?: string;
    rawTextDuration?: number;
    comparison?: TextComparisonResult;
    pageCount?: number;
}

export async function waitForServices() {
    let inceptionReady = false;
    console.log("[Services] Waiting for Inception...");
    while (!inceptionReady) {
        try {
            await axios.get(`${INCEPTION_API}/metrics`);
            inceptionReady = true;
            console.log("[Services] Inception is ready.");
        } catch (e) {
            await new Promise(r => setTimeout(r, 2000));
        }
    }
}

async function performMistralOCR(apiKey: string, buffer: Buffer, filename: string): Promise<string> {
    // Step 1: Upload file with purpose="ocr"
    const formData = new FormData();
    // @ts-ignore - Bun's global Blob may differ slightly but works
    const blob = new Blob([buffer], { type: 'application/pdf' });
    formData.append('file', blob, filename);
    formData.append('purpose', 'ocr');

    const uploadResponse = await fetch('https://api.mistral.ai/v1/files', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}` },
        body: formData,
    });

    if (!uploadResponse.ok) {
        throw new Error(`Mistral upload failed with status ${uploadResponse.status}: ${await uploadResponse.text()}`);
    }

    const uploadData = await uploadResponse.json() as { id: string };
    const fileId = uploadData.id;

    // Step 2: Process OCR with file_id
    const ocrResponse = await fetch('https://api.mistral.ai/v1/ocr', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'mistral-ocr-latest',
            document: { type: 'file', file_id: fileId }
        }),
    });

    if (!ocrResponse.ok) {
        throw new Error(`Mistral OCR failed with status ${ocrResponse.status}: ${await ocrResponse.text()}`);
    }

    const ocrData = await ocrResponse.json() as { pages: { markdown: string }[] };
    if (ocrData && ocrData.pages) {
        return ocrData.pages.map((page: any) => page.markdown || '').join('\n\n---\n\n');
    }

    return "";
}

export async function extractText(filePath: string, apiKey: string): Promise<ExtractTextResult> {
    const startTime = performance.now();
    const filename = path.basename(filePath);
    const fileExt = path.extname(filename).toLowerCase();

    const plainTextExtensions = ['.txt', '.md'];
    const pdfExtensions = ['.pdf'];

    try {
        // Plain text files - no OCR needed
        if (plainTextExtensions.includes(fileExt)) {
            const text = await fs.promises.readFile(filePath, 'utf-8');
            const duration = performance.now() - startTime;
            return { text, duration };
        }

        // For PDFs, first try to extract embedded text
        let rawText = '';
        let rawTextDuration = 0;
        let pageCount = 0;

        if (pdfExtensions.includes(fileExt)) {
            const rawStartTime = performance.now();
            const pdfResult = await extractPdfText(filePath);
            rawTextDuration = performance.now() - rawStartTime;
            rawText = pdfResult.text;
            pageCount = pdfResult.pageCount;

            if (pdfResult.success && pdfResult.charCount > 0) {
                console.log(chalk.dim(`  [PDF] Extracted ${pdfResult.charCount.toLocaleString()} chars from ${pageCount} pages (${rawTextDuration.toFixed(0)}ms)`));
            }
        }

        // Perform Mistral OCR
        const ocrStartTime = performance.now();
        const fileBuffer = await fs.promises.readFile(filePath);
        const ocrText = await performMistralOCR(apiKey, fileBuffer, filename);
        const ocrDuration = performance.now() - ocrStartTime;

        // Calculate comparison if we have raw text
        const comparison = rawText.length > 0
            ? compareTextResults(rawText, ocrText)
            : undefined;

        const totalDuration = performance.now() - startTime;

        return {
            text: ocrText,
            duration: ocrDuration,
            rawText: rawText || undefined,
            rawTextDuration: rawTextDuration || undefined,
            comparison,
            pageCount: pageCount || undefined
        };

    } catch (error) {
        const duration = performance.now() - startTime;
        console.error(chalk.red(`\n[ERROR] Processing ${filename}:`));
        if (error instanceof Error) {
            console.error(chalk.red(`  ${error.message}`));
        } else {
            console.error(chalk.red(`  ${String(error)}`));
        }
        return { text: "", duration };
    }
}

export async function saveOcrMarkdown(originalFilePath: string, ocrText: string): Promise<string> {
    // Ensure OCR directory exists
    if (!fs.existsSync(OCR_DIR)) {
        fs.mkdirSync(OCR_DIR, { recursive: true });
    }

    const originalFilename = path.basename(originalFilePath);
    const baseName = originalFilename.replace(/\.[^.]+$/, '');
    const ocrFilename = `${baseName}.ocr.md`;
    const ocrFilePath = path.join(OCR_DIR, ocrFilename);

    // Create markdown content with metadata header
    const timestamp = new Date().toISOString();
    const markdownContent = `---
source_file: ${originalFilename}
ocr_date: ${timestamp}
character_count: ${ocrText.length}
---

# OCR Output: ${originalFilename}

${ocrText}
`;

    await fs.promises.writeFile(ocrFilePath, markdownContent, 'utf-8');
    return ocrFilePath;
}

export async function getDocumentEmbedding(text: string) {
    try {
        const response = await axios.post(`${INCEPTION_API}/api/v1/embed/text`, text, {
            headers: { 'Content-Type': 'text/plain' }
        });
        return response.data;
    } catch (error) {
        if (axios.isAxiosError(error)) {
            let errorMsg = `[ERROR] Getting doc embedding: ${error.message}.`;
            if (error.response) {
                errorMsg += ` Details: ${JSON.stringify(error.response.data)}`;
            }
            throw new Error(errorMsg);
        }
        throw error;
    }
}

export async function getQueryEmbedding(text: string) {
    try {
        const response = await axios.post(`${INCEPTION_API}/api/v1/embed/query`, { text });
        return response.data;
    } catch (error) {
        console.error("[ERROR] Getting query embedding:", error);
        throw error;
    }
}

export function cosineSimilarity(vecA: number[], vecB: number[]) {
    if (!vecA || !vecB) return 0;
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
