import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import axios from 'axios'; // Keep for inception calls

const INCEPTION_API = process.env.INCEPTION_URL || 'http://localhost:8005';

export async function waitForServices() {
  let inceptionReady = false;
  console.log("Waiting for services...");
  while (!inceptionReady) {
    try {
      await axios.get(`${INCEPTION_API}/metrics`);
      inceptionReady = true;
      console.log("Inception is ready.");
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
        return ocrData.pages.map((page: any) => page.markdown || '').join('\n\n--- PAGE SEPARATOR ---\n\n');
    }

    return "";
}

export async function extractText(filePath: string, apiKey: string): Promise<{text: string, duration: number}> {
    const startTime = performance.now();
    const filename = path.basename(filePath);
    const fileExt = path.extname(filename).toLowerCase();
    
    const plainTextExtensions = ['.txt', '.md'];

    try {
        if (plainTextExtensions.includes(fileExt)) {
            const text = await fs.promises.readFile(filePath, 'utf-8');
            const duration = performance.now() - startTime;
            return { text, duration };
        }

        const fileBuffer = await fs.promises.readFile(filePath);
        const text = await performMistralOCR(apiKey, fileBuffer, filename);
        const duration = performance.now() - startTime;
        return { text, duration };

    } catch (error) {
        const duration = performance.now() - startTime;
        console.error(chalk.red(`\nError processing ${filename}:`));
        if (error instanceof Error) {
            console.error(chalk.red(`  - Error: ${error.message}`));
        } else {
            console.error(chalk.red(`  - Error: ${String(error)}`));
        }
        return { text: "", duration };
    }
}

export async function getDocumentEmbedding(text: string) {
    try {
        const response = await axios.post(`${INCEPTION_API}/api/v1/embed/text`, text, {
            headers: { 'Content-Type': 'text/plain' }
        });
        return response.data; 
    } catch (error) {
        if (axios.isAxiosError(error)) {
            let errorMsg = `Error getting doc embedding: ${error.message}.`;
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
        console.error("Error getting query embedding:", error);
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
