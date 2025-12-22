import fs from 'fs';
import path from 'path';
import chalk from 'chalk';

// --- Configuration ---
import { config } from 'dotenv';
config({ path: path.resolve(__dirname, '.env') });

const MISTRAL_API_KEY = process.env.MISTRAL_OCR_API_KEY;
const PDF_FILE_PATH = process.argv[2] || 'client/files/20060421 Complaint Parrish Medley vs Jan Wallace Re Davi Skin.pdf';

// --- Main Test Function ---
async function testMistralOcr() {
    if (!MISTRAL_API_KEY) {
        console.error(chalk.red('MISTRAL_OCR_API_KEY is not set in .env file.'));
        return;
    }

    if (!PDF_FILE_PATH) {
        console.error(chalk.red('PDF file path is required as first argument.'));
        return;
    }

    if (!fs.existsSync(PDF_FILE_PATH)) {
        console.error(chalk.red(`File not found: ${PDF_FILE_PATH}`));
        return;
    }

    console.log(chalk.blue(`\n=== Testing Mistral OCR ===`));
    console.log(chalk.cyan(`File: ${path.basename(PDF_FILE_PATH)}`));
    console.log(chalk.cyan(`Size: ${(fs.statSync(PDF_FILE_PATH).size / 1024 / 1024).toFixed(2)} MB\n`));

    try {
        const fileBuffer = fs.readFileSync(PDF_FILE_PATH);

        // --- Step 1: Upload File ---
        const formData = new FormData();
        // @ts-ignore - Bun's global Blob differs from Node's, but this works
        const blob = new Blob([fileBuffer]);
        formData.append('file', blob, path.basename(PDF_FILE_PATH));

        console.log(chalk.yellow('Step 1: Uploading file to Mistral...'));
        const uploadStart = performance.now();
        const uploadResponse = await fetch('https://api.mistral.ai/v1/files', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${MISTRAL_API_KEY}` },
            body: formData,
        });

        if (!uploadResponse.ok) {
            throw new Error(`Upload failed with status ${uploadResponse.status}: ${await uploadResponse.text()}`);
        }

        const uploadData = await uploadResponse.json() as { id: string };
        const fileId = uploadData.id;
        const uploadDuration = (performance.now() - uploadStart) / 1000;
        console.log(chalk.green(`✓ Upload successful in ${uploadDuration.toFixed(2)}s`));
        console.log(chalk.gray(`  File ID: ${fileId}\n`));

        // --- Step 2: Process OCR ---
        console.log(chalk.yellow('Step 2: Processing OCR...'));
        const ocrStart = performance.now();
        const ocrResponse = await fetch('https://api.mistral.ai/v1/document/ocr', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${MISTRAL_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'mistral-ocr-latest',
                document: { type: 'file', file_id: fileId }
            }),
        });

        if (!ocrResponse.ok) {
            throw new Error(`OCR failed with status ${ocrResponse.status}: ${await ocrResponse.text()}`);
        }

        const ocrData = await ocrResponse.json() as { pages: { markdown: string }[] };
        const ocrDuration = (performance.now() - ocrStart) / 1000;
        console.log(chalk.green(`✓ OCR successful in ${ocrDuration.toFixed(2)}s`));
        console.log(chalk.gray(`  Pages processed: ${ocrData.pages.length}`));

        // Join all pages with separator
        const fullMarkdown = ocrData.pages.map((page: any) => page.markdown || '').join('\n\n--- PAGE SEPARATOR ---\n\n');
        console.log(chalk.gray(`  Total characters: ${fullMarkdown.length}\n`));

        // Save to file for inspection
        const outputPath = path.join(path.dirname(PDF_FILE_PATH), `${path.basename(PDF_FILE_PATH, '.pdf')}_ocr.md`);
        fs.writeFileSync(outputPath, fullMarkdown);
        console.log(chalk.green(`✓ Markdown saved to: ${outputPath}\n`));

        // Show preview
        console.log(chalk.cyan('=== Preview (First 500 characters) ==='));
        console.log(fullMarkdown.substring(0, 500));
        if (fullMarkdown.length > 500) {
            console.log(chalk.gray('\n... (truncated)'));
        }

        console.log(chalk.green(`\n=== Test Complete ===`));
        console.log(chalk.gray(`Total time: ${((performance.now() - uploadStart) / 1000).toFixed(2)}s`));

    } catch (error) {
        console.error(chalk.red('\n=== Test Failed ==='));
        if (error instanceof Error) {
            console.error(chalk.red(error.message));
        } else {
            console.error(chalk.red(String(error)));
        }
        process.exit(1);
    }
}

testMistralOcr();
