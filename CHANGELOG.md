# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2025-12-24] - Search Quality & Cleanup

### Added

- **Expanded Search Results**: Increased default search results from 5 to 20
  - Configurable via `--limit <n>` option on search command
  - New `DEFAULT_SEARCH_LIMIT` constant in index.ts
- **Search Quality Test Script**: `src/test-search.ts` for comparing grep vs semantic search
  - Shows coverage comparison between text match and vector similarity
  - File-by-file comparison table with grep counts and semantic scores
  - Validates semantic search finds all files with keyword matches

### Changed

- **GPU Support Note**: Added notice that GPU version is not currently working
  - CPU version (`inception-cpu`) recommended for now
  - GPU support to be addressed in future update

### Fixed

- **Search Coverage**: All files with keyword matches now found by semantic search
  - 100% coverage of grep-matched files in top 100 semantic results
  - 81.8% precision (semantic results have actual keyword matches)

### Updated

- **.gitignore**: Added patterns for transient/generated files
  - OCR output files (`*.ocr.md`)
  - Test scripts (`client/src/test-*.ts`)
  - Preserved `.gitkeep` files in `logs/` and `ocr/` directories


## [2025-12-24] - PDF Text Comparison

### Added

- **pdf-parse Integration**: Extract embedded text from PDFs before OCR
  - Uses `pdf-parse` library (v2.4.5) for native PDF text extraction
  - Compares raw PDF text with Mistral OCR output
- **Text Comparison Statistics**:
  - Raw text character count vs OCR character count
  - Jaccard similarity percentage between raw and OCR text
  - Difference percentage for quality assessment
- **New Module**: `src/pdf-utils.ts` with text extraction and comparison utilities
  - `extractPdfText()` - Extract embedded text from PDF
  - `calculateTextSimilarity()` - Jaccard similarity on word sets
  - `compareTextResults()` - Full comparison between raw and OCR text
- **Benchmark Output**: New `[Text Comparison]` section showing:
  - Files with embedded text count
  - Total raw chars vs OCR chars
  - Average similarity and difference percentages
- **Per-File Comparison**: Files table now shows Raw/OCR/Similarity columns when data available

### Changed

- **ExtractTextResult Interface**: Enhanced to include raw text, comparison data, and page count
- **File Location**: Moved `rename-to-hash.ts` to `src/` directory


## [2025-12-24] - Privacy & Visibility Enhancements

### Changed

- **PDF File Privacy**: Renamed all PDF files to SHA256 hash filenames (16 chars)
  - Original filenames preserved in `filename-mapping.json`
  - Script `src/rename-to-hash.ts` for batch renaming
- **Keyword Highlighting**: Changed to bold yellow for better visibility
- **Benchmark Values**: All numeric values now displayed in bold for readability
- **Search Query Display**: Benchmark output now shows search query in yellow

### Added

- **filename-mapping.json**: Mapping file preserving original filenames
- **Example JSON Section**: Added complete example JSON to benchmark-format.md


## [2025-12-24] - Output Formatting Improvements

### Changed

- **Terminal Colors**: Removed problematic gray and background colors for Solarized Dark compatibility
- **Keyword Highlighting**: Changed from bgYellow.black to bold.underline for better terminal visibility
- **Benchmark Statistics**: Redesigned with horizontal two-column table layout
  - System/Sample info side by side
  - OCR/Embedding performance in parallel columns
  - Projections in compact horizontal format
- **Search Results**: Condensed to single-line file info format
- **Similarity Guide**: Simplified to plain text without gray styling

### Added

- **benchmark-format.md**: Comprehensive specification document for session JSON format
  - Full schema documentation with field descriptions
  - Instrumentation guidelines for other applications
  - Example complete session JSON
  - Comparison analysis recommendations


## [2025-12-24] - Benchmarking & Instrumentation

### Added

- **Comprehensive Benchmark System**: Full instrumentation for performance profiling across systems
  - Session JSON files saved to `logs/` with timestamp format `YYYYMMDD-HHMMSS.json`
  - System hardware info: platform, arch, CPU model, cores, memory, GPU detection
  - Per-file metrics: SHA256 hash (for privacy), size, pages, OCR time, embed time, throughput
  - Aggregated statistics: min/max/avg times, chars/sec, tokens/sec projections
  - Time projections: estimated time per 100 chars, 1000 chars, 1MB, 100MB, 1GB

- **`benchmark` Command**: Analyze and compare multiple benchmark sessions
  - Load all sessions from `logs/` folder
  - Compare performance across different systems/configurations
  - Identify fastest OCR, fastest embedding, fastest overall
  - Calculate speedup ratios between best and worst performers
  - Group analysis by unique systems
  - Generate recommendations for optimization
  - Export summary to `logs/benchmark-summary.json`

- **New TypeScript Modules**:
  - `src/types.ts`: Type definitions for SystemInfo, FileMetrics, SessionStats, BenchmarkSession
  - `src/benchmark-utils.ts`: Utility functions for system info, file hashing, statistics
  - `src/benchmark.ts`: Analyzer for comparing benchmark sessions

- **Enhanced Demo Output**:
  - System information display at start (platform, CPU, memory, GPU)
  - Session ID tracking for each run
  - Detailed per-file metrics table (hash, size, pages, chars, OCR time, embed time, chunks)
  - Time projections section with estimated processing times at scale
  - Automatic session save to `logs/` directory

- **File Hash Privacy**: Uses SHA256 hash (first 16 chars) instead of filenames in session logs

### Changed

- **Default PDF Count**: Changed from 3 to 1 for faster first-run experience
- **Demo Output**: Added comprehensive benchmark statistics section at end
- **docker-compose.yml**: Added `logs/` volume mount for session persistence


## [2025-12-24] - Platform Detection & OCR Export

### Added

- **Platform Detection**: Automatic detection of macOS ARM64 (Apple Silicon M1/M2/M3), Linux ARM64, and NVIDIA GPU in test scripts
- **`start-inception.sh`**: New standalone script to start the appropriate Inception backend based on detected platform
  - Supports manual override with `--gpu` or `--cpu` flags
  - Checks for existing running services before starting new ones
- **`--pdf-count` Option**: New CLI option to limit number of files to index
  - Default: 3 random files in demo mode
  - Use `--pdf-count 0` to index all files
  - Works with both `index` and `demo` commands
- **OCR Markdown Export**: OCR output now saved as `.ocr.md` files in `client/ocr/` directory
  - Includes YAML frontmatter with source file, date, and character count
- **Similarity Score Guide**: Search output now explains what similarity scores mean
  - Strong match (> 0.7), Moderate match (0.5-0.7), Weak match (< 0.5)
  - Labels showing `(max)`, `(min)`, `(avg)` for each result
- **File Metadata in Search Results**: Each match now displays file path, file size, and character count
- **Keyword Highlighting**: Matched query terms are now highlighted (bold yellow background) in snippets
- **Benchmark Statistics**: Detailed timing metrics for OCR, embedding, and search operations
  - Total time, OCR time, embedding time with percentages
  - Average time per file and per chunk
  - Search throughput (chunks/sec)
- **`.env.example`**: New template file documenting all environment variables with defaults

### Changed

- **Environment Configuration**: Consolidated from `client/.env` to single root `.env` file
- **`docker-compose.yml`**: Updated to reference root `.env` and added `client/ocr` volume mount
- **`test-docker-stack.sh`**: Complete rewrite with platform detection and service checking
  - Uses `PROFILE_OVERRIDE` instead of `PROFILE` for manual override
  - Skips service startup if Inception is already running
  - Text-based status indicators replace emojis
- **CLI Output Format**: Replaced all emoji icons with text-based labels
  - `[OK]`, `[FAIL]`, `[ERROR]`, `[INFO]`, `[SKIP]` status indicators
  - Cleaner, more accessible terminal output
- **README.md**: Major revision with improved formatting
  - Added Quickstart section with three options immediately after introduction
  - Fixed grammar and spelling throughout
  - Removed all emojis
  - Added tables for environment variables and performance metrics
  - Updated file structure documentation

### Fixed

- Page separator in OCR output changed from `--- PAGE SEPARATOR ---` to cleaner `---`
- Improved regex escaping for keyword highlighting to handle special characters

### Removed

- Duplicate `client/.env` file (now using root `.env` only)
- Emoji icons from all CLI output and documentation


## [Initial Release]

### Added

- Initial project setup
- `docker-compose.yml` with profiles for `m1` (CPU) and `gpu` (CUDA)
- Bun/TypeScript client application
- `api.ts`: Integration with Inception and Mistral OCR APIs
- `index.ts`: CLI with `index`, `search`, `run`, and `demo` commands
- In-memory vector search using Cosine Similarity
- Support for chunked embeddings from Inception
- `README.md` with documentation
