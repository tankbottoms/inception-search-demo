import fs from 'fs';
import path from 'path';
import chalk from 'chalk';

// --- Configuration ---
// Load .env from the project root (one level up from tests/)
import { config } from 'dotenv';

// Find the project root by looking for docker-compose.yml
function findProjectRoot(): string {
    let dir = __dirname;
    for (let i = 0; i < 5; i++) {
        if (fs.existsSync(path.join(dir, 'docker-compose.yml'))) {
            return dir;
        }
        dir = path.dirname(dir);
    }
    return process.cwd();
}

const PROJECT_ROOT = findProjectRoot();
config({ path: path.join(PROJECT_ROOT, '.env') });

const MISTRAL_API_KEY = process.env.MISTRAL_OCR_API_KEY;
const DEFAULT_PDF = path.join(PROJECT_ROOT, 'client/files/20060421 Complaint Parrish Medley vs Jan Wallace Re Davi Skin.pdf');
const PDF_FILE_PATH = process.argv[2] || DEFAULT_PDF;

// --- Main Test Function ---
async function testMistralOcr() {
    console.log(chalk.blue(`\n=== Testing Mistral OCR API ===`));
    console.log(chalk.gray(`Project root: ${PROJECT_ROOT}`));
    console.log(chalk.gray(`Looking for .env at: ${path.join(PROJECT_ROOT, '.env')}\n`));

    if (!MISTRAL_API_KEY) {
        console.error(chalk.red('MISTRAL_OCR_API_KEY is not set.'));
        console.error(chalk.yellow('\nPlease create a .env file in the project root with:'));
        console.error(chalk.cyan('  MISTRAL_OCR_API_KEY=your-api-key-here'));
        console.error(chalk.yellow('\nGet your API key at: https://console.mistral.ai/'));
        process.exit(1);
    }

    if (!fs.existsSync(PDF_FILE_PATH)) {
        console.error(chalk.red(`File not found: ${PDF_FILE_PATH}`));
        process.exit(1);
    }

    console.log(chalk.cyan(`File: ${path.basename(PDF_FILE_PATH)}`));
    console.log(chalk.cyan(`Size: ${(fs.statSync(PDF_FILE_PATH).size / 1024 / 1024).toFixed(2)} MB\n`));

    const totalStart = performance.now();
    try {
        const fileBuffer = fs.readFileSync(PDF_FILE_PATH);

        // --- Step 1: Upload file to Mistral with purpose="ocr" ---
        console.log(chalk.yellow('Step 1: Uploading file to Mistral...'));
        const formData = new FormData();
        // @ts-ignore - Bun's Blob API
        const blob = new Blob([fileBuffer], { type: 'application/pdf' });
        formData.append('file', blob, path.basename(PDF_FILE_PATH));
        formData.append('purpose', 'ocr');

        const uploadStart = performance.now();
        const uploadResponse = await fetch('https://api.mistral.ai/v1/files', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${MISTRAL_API_KEY}`
            },
            body: formData,
        });

        if (!uploadResponse.ok) {
            throw new Error(`Upload failed with status ${uploadResponse.status}: ${await uploadResponse.text()}`);
        }

        const uploadData = await uploadResponse.json() as { id: string };
        const fileId = uploadData.id;
        const uploadDuration = (performance.now() - uploadStart) / 1000;
        console.log(chalk.green(`OK Upload successful in ${uploadDuration.toFixed(2)}s`));
        console.log(chalk.gray(`  File ID: ${fileId}\n`));

        // --- Step 2: Process OCR ---
        console.log(chalk.yellow('Step 2: Processing OCR with Mistral...'));
        const ocrStart = performance.now();
        const ocrResponse = await fetch('https://api.mistral.ai/v1/ocr', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${MISTRAL_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'mistral-ocr-latest',
                document: {
                    type: 'file',
                    file_id: fileId
                }
            }),
        });

        if (!ocrResponse.ok) {
            throw new Error(`OCR failed with status ${ocrResponse.status}: ${await ocrResponse.text()}`);
        }

        const ocrData = await ocrResponse.json() as { pages: { markdown: string }[] };
        const ocrDuration = (performance.now() - ocrStart) / 1000;
        console.log(chalk.green(`OK OCR successful in ${ocrDuration.toFixed(2)}s`));
        console.log(chalk.gray(`  Pages processed: ${ocrData.pages.length}`));

        // Join all pages with separator
        const fullMarkdown = ocrData.pages.map((page: any) => page.markdown || '').join('\n\n--- PAGE SEPARATOR ---\n\n');
        console.log(chalk.gray(`  Total characters: ${fullMarkdown.length}\n`));

        // Save to file for inspection
        const outputPath = path.join(path.dirname(PDF_FILE_PATH), `${path.basename(PDF_FILE_PATH, '.pdf')}_ocr.md`);
        fs.writeFileSync(outputPath, fullMarkdown);
        console.log(chalk.green(`OK Markdown saved to: ${outputPath}\n`));

        // Show preview
        console.log(chalk.cyan('=== Preview (First 500 characters) ==='));
        console.log(fullMarkdown.substring(0, 500));
        if (fullMarkdown.length > 500) {
            console.log(chalk.gray('\n... (truncated)'));
        }

        console.log(chalk.green(`\n=== Test Complete - SUCCESS ===`));
        console.log(chalk.gray(`Total time: ${((performance.now() - totalStart) / 1000).toFixed(2)}s`));

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
