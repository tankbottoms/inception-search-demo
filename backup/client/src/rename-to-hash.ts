#!/usr/bin/env bun
/**
 * Rename PDF files to their SHA256 hash for privacy
 * Usage: bun run rename-to-hash.ts [directory]
 */

import { readdirSync, readFileSync, renameSync, existsSync } from 'fs';
import { join, extname, basename } from 'path';
import crypto from 'crypto';

const directory = process.argv[2] || 'files';

if (!existsSync(directory)) {
    console.error(`Directory not found: ${directory}`);
    process.exit(1);
}

const files = readdirSync(directory).filter(f => extname(f).toLowerCase() === '.pdf');

console.log(`Found ${files.length} PDF files in ${directory}\n`);

const renamed: { original: string; hash: string }[] = [];

for (const file of files) {
    const filePath = join(directory, file);
    const buffer = readFileSync(filePath);
    const hash = crypto.createHash('sha256').update(buffer).digest('hex').substring(0, 16);
    const newName = `${hash}.pdf`;
    const newPath = join(directory, newName);

    if (file === newName) {
        console.log(`[SKIP] ${file} (already hashed)`);
        continue;
    }

    if (existsSync(newPath)) {
        console.log(`[SKIP] ${file} -> ${newName} (target exists, possible duplicate)`);
        continue;
    }

    renameSync(filePath, newPath);
    renamed.push({ original: file, hash: newName });
    console.log(`[OK] ${file}`);
    console.log(`     -> ${newName}\n`);
}

console.log(`\nRenamed ${renamed.length} files`);

// Output mapping for reference
if (renamed.length > 0) {
    const mappingPath = join(directory, 'filename-mapping.json');
    const mapping = renamed.reduce((acc, { original, hash }) => {
        acc[hash] = original;
        return acc;
    }, {} as Record<string, string>);

    Bun.write(mappingPath, JSON.stringify(mapping, null, 2));
    console.log(`Mapping saved to: ${mappingPath}`);
}
