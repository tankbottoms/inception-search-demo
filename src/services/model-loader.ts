/**
 * ONNX Model Loader with resolution chain:
 * 1. Check local cache
 * 2. Check HuggingFace for ONNX model
 * 3. Fallback to Python converter service
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'fs';
import { join, basename } from 'path';
import * as ort from 'onnxruntime-node';
import type { ModelEntry, ModelRegistry, OnnxSessionOptions } from '../types';
import { logger, Timer } from './logger';
import { MODEL_CACHE_DIR, settings } from './config';
import { getExecutionProviders, detectHardware } from './hardware';

// Constants
const REGISTRY_PATH = join(MODEL_CACHE_DIR, 'registry.json');
const HUGGINGFACE_BASE = 'https://huggingface.co';
const CONVERTER_URL = process.env.CONVERTER_URL || 'http://localhost:8010';

interface ModelFiles {
  modelPath: string;
  tokenizerPath: string;
  configPath?: string;
}

interface LoadedModel {
  session: ort.InferenceSession;
  tokenizer: TokenizerConfig;
  config: ModelEntry;
}

interface TokenizerConfig {
  vocab: Record<string, number>;
  merges?: string[];
  added_tokens?: Array<{ id: number; content: string }>;
  model_max_length?: number;
  // Pre-parsed special tokens
  cls_token_id?: number;
  sep_token_id?: number;
  pad_token_id?: number;
  unk_token_id?: number;
}

// Model cache
const modelCache = new Map<string, LoadedModel>();

/**
 * Load model registry from disk
 */
export function loadRegistry(): ModelRegistry {
  try {
    const content = readFileSync(REGISTRY_PATH, 'utf-8');
    return JSON.parse(content) as ModelRegistry;
  } catch (error) {
    logger.warn('Could not load registry, using defaults', { error: String(error) });
    return {
      version: '1.0',
      cache_dir: MODEL_CACHE_DIR,
      models: [],
    };
  }
}

/**
 * Get a model entry from the registry
 */
export function getModelEntry(modelId: string): ModelEntry | undefined {
  const registry = loadRegistry();
  return registry.models.find(m => m.id === modelId);
}

/**
 * Get local model path for a given model entry
 */
function getLocalModelPath(entry: ModelEntry): string {
  // Convert HuggingFace name to local path
  // e.g., "freelawproject/modernbert-embed-base_finetune_512" -> "freelawproject--modernbert-embed-base_finetune_512"
  const safeName = entry.name.replace(/\//g, '--');
  return join(MODEL_CACHE_DIR, safeName);
}

/**
 * Check if model exists locally with required files
 */
function checkLocalModel(entry: ModelEntry): ModelFiles | null {
  const modelDir = getLocalModelPath(entry);

  if (!existsSync(modelDir)) {
    return null;
  }

  // Check for ONNX model file
  const files = readdirSync(modelDir);
  const onnxFile = files.find(f => f.endsWith('.onnx'));

  if (!onnxFile) {
    return null;
  }

  const modelPath = join(modelDir, onnxFile);
  const tokenizerPath = join(modelDir, 'tokenizer.json');

  if (!existsSync(tokenizerPath)) {
    logger.warn(`Model found but tokenizer.json missing: ${modelDir}`);
    return null;
  }

  return {
    modelPath,
    tokenizerPath,
    configPath: existsSync(join(modelDir, 'config.json'))
      ? join(modelDir, 'config.json')
      : undefined,
  };
}

/**
 * Try to download ONNX model from HuggingFace
 */
async function downloadFromHuggingFace(entry: ModelEntry): Promise<ModelFiles | null> {
  const timer = new Timer('download-hf');
  const modelDir = getLocalModelPath(entry);

  logger.info(`Attempting to download ONNX model from HuggingFace: ${entry.name}`);

  // Required files to download
  const requiredFiles = ['model.onnx', 'tokenizer.json'];
  const optionalFiles = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json'];

  try {
    // Ensure directory exists
    mkdirSync(modelDir, { recursive: true });

    // Try to download from onnx branch first, then main
    const branches = ['onnx', 'main'];
    let downloadedModel = false;

    for (const branch of branches) {
      if (downloadedModel) break;

      for (const file of requiredFiles) {
        const url = `${HUGGINGFACE_BASE}/${entry.name}/resolve/${branch}/${file}`;
        const localPath = join(modelDir, file);

        try {
          logger.debug(`Downloading ${url}`);
          const response = await fetch(url);

          if (!response.ok) {
            if (file === 'model.onnx') {
              // Try alternative ONNX file names
              const altNames = ['model_quantized.onnx', 'model_optimized.onnx', 'onnx/model.onnx'];
              for (const altName of altNames) {
                const altUrl = `${HUGGINGFACE_BASE}/${entry.name}/resolve/${branch}/${altName}`;
                const altResponse = await fetch(altUrl);
                if (altResponse.ok) {
                  const buffer = await altResponse.arrayBuffer();
                  writeFileSync(localPath, Buffer.from(buffer));
                  downloadedModel = true;
                  logger.info(`Downloaded ${altName} as model.onnx`);
                  break;
                }
              }
              if (!downloadedModel) {
                throw new Error(`Failed to download ${file}: ${response.status}`);
              }
            } else {
              throw new Error(`Failed to download ${file}: ${response.status}`);
            }
          } else {
            const buffer = await response.arrayBuffer();
            writeFileSync(localPath, Buffer.from(buffer));
            if (file === 'model.onnx') downloadedModel = true;
            logger.debug(`Downloaded ${file}`);
          }
        } catch (error) {
          if (branch === branches[branches.length - 1]) {
            logger.warn(`Failed to download ${file} from any branch`, { error: String(error) });
            return null;
          }
          break; // Try next branch
        }
      }
    }

    // Download optional files (don't fail if missing)
    for (const file of optionalFiles) {
      const url = `${HUGGINGFACE_BASE}/${entry.name}/resolve/main/${file}`;
      const localPath = join(modelDir, file);

      try {
        const response = await fetch(url);
        if (response.ok) {
          const buffer = await response.arrayBuffer();
          writeFileSync(localPath, Buffer.from(buffer));
        }
      } catch {
        // Ignore optional file failures
      }
    }

    timer.log('HuggingFace download complete');

    return {
      modelPath: join(modelDir, 'model.onnx'),
      tokenizerPath: join(modelDir, 'tokenizer.json'),
      configPath: existsSync(join(modelDir, 'config.json'))
        ? join(modelDir, 'config.json')
        : undefined,
    };
  } catch (error) {
    logger.error('Failed to download from HuggingFace', error);
    return null;
  }
}

/**
 * Request model conversion from Python converter service
 */
async function convertWithPython(entry: ModelEntry): Promise<ModelFiles | null> {
  const timer = new Timer('convert-python');

  logger.info(`Requesting model conversion from Python service: ${entry.name}`);

  try {
    const response = await fetch(`${CONVERTER_URL}/convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: entry.name,
        output_dir: MODEL_CACHE_DIR,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Converter service returned ${response.status}: ${error}`);
    }

    const result = await response.json() as { success: boolean; output_path?: string; error?: string };

    if (!result.success) {
      throw new Error(result.error || 'Conversion failed');
    }

    timer.log('Python conversion complete');

    // Check if files exist now
    return checkLocalModel(entry);
  } catch (error) {
    logger.error('Python conversion failed', error);
    return null;
  }
}

/**
 * Resolve model files using the resolution chain
 */
async function resolveModelFiles(entry: ModelEntry): Promise<ModelFiles> {
  const timer = new Timer('resolve-model');

  // Step 1: Check local cache
  logger.info(`Checking local cache for model: ${entry.id}`);
  let files = checkLocalModel(entry);
  if (files) {
    logger.info(`Found model in local cache: ${files.modelPath}`);
    timer.log('Model found in cache');
    return files;
  }

  // Step 2: Try HuggingFace ONNX download
  logger.info(`Model not in cache, trying HuggingFace ONNX download`);
  files = await downloadFromHuggingFace(entry);
  if (files) {
    timer.log('Downloaded from HuggingFace');
    return files;
  }

  // Step 3: Fall back to Python converter
  logger.info(`HuggingFace download failed, trying Python converter`);
  files = await convertWithPython(entry);
  if (files) {
    timer.log('Converted with Python');
    return files;
  }

  throw new Error(
    `Failed to resolve model files for ${entry.id}. ` +
    `Tried: local cache, HuggingFace ONNX, Python converter.`
  );
}

/**
 * Load tokenizer configuration
 */
function loadTokenizer(tokenizerPath: string): TokenizerConfig {
  const content = readFileSync(tokenizerPath, 'utf-8');
  const raw = JSON.parse(content);

  // Handle Hugging Face tokenizers.json format
  const config: TokenizerConfig = {
    vocab: {},
    model_max_length: raw.truncation?.max_length || 512,
  };

  // Extract vocabulary
  if (raw.model?.vocab) {
    config.vocab = raw.model.vocab;
  } else if (raw.vocab) {
    config.vocab = raw.vocab;
  }

  // Extract merges for BPE tokenizers
  if (raw.model?.merges) {
    config.merges = raw.model.merges;
  }

  // Extract added tokens
  if (raw.added_tokens) {
    config.added_tokens = raw.added_tokens;

    // Find special token IDs
    for (const token of raw.added_tokens) {
      if (token.content === '[CLS]' || token.content === '<s>') {
        config.cls_token_id = token.id;
      }
      if (token.content === '[SEP]' || token.content === '</s>') {
        config.sep_token_id = token.id;
      }
      if (token.content === '[PAD]' || token.content === '<pad>') {
        config.pad_token_id = token.id;
      }
      if (token.content === '[UNK]' || token.content === '<unk>') {
        config.unk_token_id = token.id;
      }
    }
  }

  return config;
}

/**
 * Check if CUDA provider is available in ONNX Runtime
 */
async function checkOnnxCudaAvailable(): Promise<boolean> {
  try {
    // Check if onnxruntime-gpu is available by trying to get available providers
    // onnxruntime-node includes CUDA support on systems with CUDA installed
    const hw = detectHardware();

    if (hw.provider !== 'cuda') {
      return false;
    }

    // The actual CUDA availability is validated when creating a session
    // We return true if hardware detection found CUDA
    return true;
  } catch {
    return false;
  }
}

/**
 * Create ONNX session options with CUDA support
 */
function createSessionOptions(preferCuda: boolean = true): ort.InferenceSession.SessionOptions {
  const hw = detectHardware();
  const numCpus = require('os').cpus().length;

  const options: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    logSeverityLevel: 2, // Warning
  };

  // Try CUDA if hardware supports it and preference is set
  if (hw.provider === 'cuda' && preferCuda) {
    // Use CUDA with CPU fallback
    options.executionProviders = [
      { name: 'cuda', deviceId: 0 },
      'cpu',
    ];
    logger.info(`ONNX session options: CUDA (device 0) with CPU fallback`);
  } else {
    // CPU only
    options.executionProviders = ['cpu'];
    options.intraOpNumThreads = Math.max(1, Math.floor(numCpus / 2));
    options.interOpNumThreads = Math.max(1, Math.floor(numCpus / 4));
    logger.info(`ONNX session options: CPU with ${options.intraOpNumThreads} intra-op threads`);
  }

  return options;
}

/**
 * Load a model by ID
 */
export async function loadModel(modelId: string): Promise<LoadedModel> {
  // Check cache first
  const cached = modelCache.get(modelId);
  if (cached) {
    logger.debug(`Using cached model: ${modelId}`);
    return cached;
  }

  const timer = new Timer('load-model');

  // Get model entry from registry
  const entry = getModelEntry(modelId);
  if (!entry) {
    throw new Error(`Model not found in registry: ${modelId}`);
  }

  if (!entry.enabled) {
    throw new Error(`Model is disabled: ${modelId}`);
  }

  // Resolve model files
  const files = await resolveModelFiles(entry);

  // Load tokenizer
  logger.info(`Loading tokenizer from ${files.tokenizerPath}`);
  const tokenizer = loadTokenizer(files.tokenizerPath);
  timer.checkpoint('tokenizer-loaded');

  // Create ONNX session with CUDA preference
  logger.info(`Creating ONNX session for ${files.modelPath}`);

  let session: ort.InferenceSession;
  let usedProvider: 'cuda' | 'cpu' = 'cpu';

  try {
    // Try with CUDA first
    const options = createSessionOptions(true);
    session = await ort.InferenceSession.create(files.modelPath, options);
    usedProvider = detectHardware().provider === 'cuda' ? 'cuda' : 'cpu';
    timer.checkpoint('session-created');
  } catch (cudaError) {
    // CUDA failed, fall back to CPU
    logger.warn('CUDA session creation failed, falling back to CPU', { error: String(cudaError) });

    try {
      const cpuOptions = createSessionOptions(false);
      session = await ort.InferenceSession.create(files.modelPath, cpuOptions);
      usedProvider = 'cpu';
      timer.checkpoint('session-created-cpu-fallback');
    } catch (cpuError) {
      logger.error('Failed to create ONNX session with both CUDA and CPU', cpuError);
      throw cpuError;
    }
  }

  logger.info(`Model loaded successfully`, {
    modelId,
    provider: usedProvider,
    inputNames: session.inputNames,
    outputNames: session.outputNames,
  });

  const loaded: LoadedModel = {
    session,
    tokenizer,
    config: entry,
  };

  // Cache the model
  modelCache.set(modelId, loaded);

  timer.log('Model loading complete');

  return loaded;
}

/**
 * Check if a model is available (without loading it)
 */
export async function checkModelAvailable(modelId: string): Promise<{
  available: boolean;
  location?: 'cache' | 'huggingface' | 'converter';
  path?: string;
  error?: string;
}> {
  const entry = getModelEntry(modelId);
  if (!entry) {
    return { available: false, error: 'Model not in registry' };
  }

  // Check local cache
  const localFiles = checkLocalModel(entry);
  if (localFiles) {
    return { available: true, location: 'cache', path: localFiles.modelPath };
  }

  // Check HuggingFace (without downloading)
  const hfUrl = `${HUGGINGFACE_BASE}/${entry.name}/resolve/main/model.onnx`;
  try {
    const response = await fetch(hfUrl, { method: 'HEAD' });
    if (response.ok) {
      return { available: true, location: 'huggingface' };
    }
  } catch {
    // Continue to converter check
  }

  // Check converter service availability
  try {
    const response = await fetch(`${CONVERTER_URL}/health`, { method: 'GET' });
    if (response.ok) {
      return { available: true, location: 'converter' };
    }
  } catch {
    // Converter not available
  }

  return { available: false, error: 'Model not found and no conversion service available' };
}

/**
 * Unload a model from cache
 */
export function unloadModel(modelId: string): boolean {
  const model = modelCache.get(modelId);
  if (model) {
    // Release session resources
    model.session.release();
    modelCache.delete(modelId);
    logger.info(`Unloaded model: ${modelId}`);
    return true;
  }
  return false;
}

/**
 * Get list of loaded models
 */
export function getLoadedModels(): string[] {
  return Array.from(modelCache.keys());
}

// Export types
export type { LoadedModel, TokenizerConfig, ModelFiles };
