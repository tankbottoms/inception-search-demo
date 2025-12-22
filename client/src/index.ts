import { Command } from 'commander';
import { glob } from 'glob';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { waitForServices, extractText, getDocumentEmbedding, getQueryEmbedding, cosineSimilarity } from './api.js';

const program = new Command();
const DB_FILE = 'embeddings.json';

interface IndexedDoc {
    id: string;
    text: string;
    embedding: number[];
}

function formatFileSize(bytes: number): string {
    if (!bytes && bytes !== 0) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

function createSnippet(text: string, query: string, snippetLength = 200): string {
    const queryWords = query.toLowerCase().split(/\s+/).filter(Boolean);
    const lowerText = text.toLowerCase();
    
    let firstMatchIndex = -1;
    for (const word of queryWords) {
        const index = lowerText.indexOf(word);
        if (index !== -1) {
            firstMatchIndex = index;
            break;
        }
    }

    if (firstMatchIndex === -1) {
        return text.length > snippetLength ? text.substring(0, snippetLength) + '...' : text;
    }

    const start = Math.max(0, firstMatchIndex - Math.round(snippetLength / 3));
    let snippet = text.substring(start);
    
    for (const word of queryWords) {
        const regex = new RegExp(`(${word})`, 'gi');
        snippet = snippet.replace(regex, chalk.yellow.bold('$1'));
    }

    if (snippet.length > snippetLength) {
        snippet = snippet.substring(0, snippetLength) + '...';
    }

    if (start > 0) {
        snippet = '...' + snippet;
    }

    return snippet.replace(/\n/g, ' ');
}

function isLikelyGarbage(text: string, threshold = 0.5): boolean {
    if (!text) return false;
    let nonAscii = 0;
    for (let i = 0; i < text.length; i++) {
        if (text.charCodeAt(i) > 255) {
            nonAscii++;
        }
    }
    return (nonAscii / text.length) > threshold;
}

async function indexFiles(filePathOrDir: string) {
    const mistralApiKey = process.env.MISTRAL_OCR_API_KEY;
    if (!mistralApiKey) {
        console.error(chalk.red("MISTRAL_OCR_API_KEY environment variable not set."));
        return;
    }

    await waitForServices();
    console.log(chalk.blue(`\nIndexing from: ${filePathOrDir}`));

    let files: string[];
    if (fs.statSync(filePathOrDir).isDirectory()) {
        files = await glob(`${filePathOrDir}/**/*.{pdf,png,jpg,jpeg,gif,bmp,tiff,docx,pptx,txt,md}`);
    } else {
        files = [filePathOrDir];
    }
    
    const db: IndexedDoc[] = [];
    console.log(chalk.blue(`Found ${files.length} file(s) to process.`));

    for (const [index, file] of files.entries()) {
        const fileSize = fs.statSync(file).size;
        console.log(chalk.cyan(`\n[${index + 1}/${files.length}] Processing ${path.basename(file)} (${formatFileSize(fileSize)})`));

        const ocrSpinner = ora('OCR with Mistral...').start();
        const { text, duration: ocrDuration } = await extractText(file, mistralApiKey);
        if (!text) {
            ocrSpinner.fail("OCR with Mistral failed (no text extracted).");
            continue;
        }
        ocrSpinner.succeed(`OCR with Mistral done in ${(ocrDuration / 1000).toFixed(2)}s. Extracted ${text.length} chars.`);

        if (isLikelyGarbage(text)) {
            console.log(chalk.red("  - Skipped: Text appears to be garbled/corrupt."));
            continue;
        }

        const MAX_TEXT_LENGTH = 10000000;
        if (text.length > MAX_TEXT_LENGTH) {
            console.log(chalk.red(`  - Skipped: Text too long (${text.length} > ${MAX_TEXT_LENGTH})`));
            continue;
        }

        const embedSpinner = ora('Embedding with Inception...').start();
        const embedStartTime = performance.now();
        try {
            const embeddingResponse = await getDocumentEmbedding(text);
            const embedDuration = performance.now() - embedStartTime;

            if (embeddingResponse && embeddingResponse.embeddings && Array.isArray(embeddingResponse.embeddings)) {
                const numChunks = embeddingResponse.embeddings.length;
                embedSpinner.succeed(`Embedding with Inception done in ${(embedDuration / 1000).toFixed(2)}s. Got ${numChunks} chunks.`);
                for (const item of embeddingResponse.embeddings) {
                    if (item.embedding && Array.isArray(item.embedding)) {
                        db.push({
                            id: `${path.basename(file)}#${item.chunk_number}`,
                            text: item.chunk,
                            embedding: item.embedding
                        });
                    }
                }
            } else {
                 embedSpinner.fail("Embedding with Inception failed (unexpected response).");
            }
        } catch (e) {
            const embedDuration = performance.now() - embedStartTime;
            embedSpinner.fail(`Embedding with Inception failed in ${(embedDuration / 1000).toFixed(2)}s.`);
        }
    }

    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));
    console.log(chalk.green(`\n\nSuccessfully indexed ${db.length} text chunks. Saved to ${DB_FILE}`));
}

async function search(query: string) {
    if (!fs.existsSync(DB_FILE)) {
        console.log(chalk.red(`No index found. Run 'index' command first.`));
        return;
    }
    const db: IndexedDoc[] = JSON.parse(fs.readFileSync(DB_FILE, 'utf-8'));
    
    await waitForServices();
    console.log(chalk.blue(`\n🔍 Searching for: "${query}"`));

    const queryEmbResponse = await getQueryEmbedding(query);
    let queryEmb: number[] = [];
     if (Array.isArray(queryEmbResponse)) {
        queryEmb = queryEmbResponse;
    } else if (queryEmbResponse.embedding) {
        queryEmb = queryEmbResponse.embedding;
    }

    if (queryEmb.length === 0) {
        console.error(chalk.red("Failed to get query embedding"));
        return;
    }

    const results = db.map(doc => ({
        doc,
        score: cosineSimilarity(queryEmb, doc.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

    if (results.length === 0) {
        console.log(chalk.yellow("No results found."));
        return;
    }

    const scores = results.map(r => r.score);
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    console.log(chalk.green(`\n📊 Search Statistics (Top ${results.length}):`));
    console.log(`  Max Similarity: ${maxScore.toFixed(4)} (Min Distance: ${(1-maxScore).toFixed(4)})`);
    console.log(`  Min Similarity: ${minScore.toFixed(4)} (Max Distance: ${(1-minScore).toFixed(4)})`);
    console.log(`  Avg Similarity: ${avgScore.toFixed(4)} (Avg Distance: ${(1-avgScore).toFixed(4)})`);

    console.log(chalk.green(`\n🎯 Top Relevant Matches:`));

    results.forEach((res, i) => {
        const [filename] = res.doc.id.split('#');
        const snippet = createSnippet(res.doc.text, query);
        console.log(chalk.cyan(`--- Match ${i+1} ---`));
        console.log(`  ${chalk.bold('File:')}      ${filename}`);
        console.log(`  ${chalk.bold('Similarity:')} ${res.score.toFixed(4)}`);
        console.log(`  ${chalk.bold('Snippet:')}   "${snippet}"`);
        console.log('');
    });
}

program
    .name('inception-client')
    .description('CLI to demo Inception with Mistral OCR for document indexing and semantic search')
    .version('1.0.0');

program.command('index')
    .argument('<path>', 'File or directory to index')
    .description('Index document(s) and generate embeddings')
    .action(indexFiles);

program.command('search')
    .argument('<query>', 'Search query')
    .description('Search indexed documents using semantic search')
    .action(search);

program.command('run')
    .argument('<file>', 'File to index')
    .argument('<query>', 'Search query')
    .description('Index a single file and search it')
    .action(async (file: string, query: string) => {
        // Resolve file path
        const sourcePath = path.resolve(file);

        if (!fs.existsSync(sourcePath)) {
            console.error(chalk.red(`File not found: ${file}`));
            return;
        }

        // Copy file to files/ directory if not already there
        const filesDir = path.join(process.cwd(), 'files');
        if (!fs.existsSync(filesDir)) {
            fs.mkdirSync(filesDir, { recursive: true });
        }

        const fileName = path.basename(file);
        const destPath = path.join(filesDir, fileName);

        // Copy file if it's not already in files/
        if (path.resolve(sourcePath) !== path.resolve(destPath)) {
            console.log(chalk.blue(`Copying ${fileName} to files/ directory...`));
            fs.copyFileSync(sourcePath, destPath);
            console.log(chalk.green(`✓ File copied to ${destPath}\n`));
        }

        // Index and search
        await indexFiles(destPath);
        await search(query);
    });

program.command('demo')
    .argument('[file]', 'Optional: File to add to demo')
    .argument('[query]', 'Optional: Search query (default: "securities fraud")')
    .description('Run demo: index all files and search')
    .action(async (file?: string, query?: string) => {
        const filesDir = path.join(process.cwd(), 'files');

        // If a file is provided, copy it to files/
        if (file) {
            const sourcePath = path.resolve(file);
            if (!fs.existsSync(sourcePath)) {
                console.error(chalk.red(`File not found: ${file}`));
                return;
            }

            if (!fs.existsSync(filesDir)) {
                fs.mkdirSync(filesDir, { recursive: true });
            }

            const fileName = path.basename(file);
            const destPath = path.join(filesDir, fileName);

            if (path.resolve(sourcePath) !== path.resolve(destPath)) {
                console.log(chalk.blue(`Adding ${fileName} to demo files...`));
                fs.copyFileSync(sourcePath, destPath);
                console.log(chalk.green(`✓ File added to ${destPath}\n`));
            }
        }

        // Index all files in files/ directory
        if (!fs.existsSync(filesDir)) {
            console.error(chalk.red(`Files directory not found: ${filesDir}`));
            return;
        }

        console.log(chalk.blue.bold('\n=== Running Inception Demo ===\n'));
        await indexFiles(filesDir);

        // Search with provided query or default
        const searchQuery = query || "securities fraud";
        console.log(chalk.blue.bold(`\n=== Searching for: "${searchQuery}" ===\n`));
        await search(searchQuery);
    });

program.parse();
