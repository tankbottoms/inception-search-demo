/**
 * HunyuanOCR - Local ONNX-based OCR inference
 *
 * Supports both CPU and CUDA execution providers via ONNX Runtime.
 * Model is automatically converted from PyTorch on first use.
 */

import { existsSync, readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { logger, Timer } from '../logger';
import { detectHardware } from '../hardware';
import { MODEL_CACHE_DIR } from '../config';

// Model paths
const MODEL_DIR = join(MODEL_CACHE_DIR, 'tencent--HunyuanOCR');
const PYTORCH_MODEL_DIR = join(MODEL_CACHE_DIR, 'tencent--HunyuanOCR-pytorch');
const ONNX_MODEL_DIR = join(MODEL_CACHE_DIR, 'tencent--HunyuanOCR-onnx');

// Default preprocessing config
const DEFAULT_PREPROCESS_CONFIG = {
  imageSize: 448,
  patchSize: 16,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  maxImageSize: 2048,
};

export interface HunyuanOCRResult {
  text: string;
  confidence?: number;
  processingTime: number;
  provider: 'cpu' | 'cuda';
}

export interface HunyuanOCROptions {
  maxTokens?: number;
  language?: string;
  detailed?: boolean;
}

interface PreprocessConfig {
  imageSize: number;
  patchSize: number;
  mean: number[];
  std: number[];
  maxImageSize: number;
}

interface ModelConfig {
  vision: {
    hiddenSize: number;
    numAttentionHeads: number;
    patchSize: number;
    imageSize: number;
  };
  text: {
    hiddenSize: number;
    vocabSize: number;
  };
  specialTokens: {
    bosTokenId: number;
    eosTokenId: number;
    padTokenId: number;
    imageTokenId: number;
  };
}

// Cached sessions
let visionEncoderSession: ort.InferenceSession | null = null;
let textDecoderSession: ort.InferenceSession | null = null;
let preprocessConfig: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG;
let modelConfig: ModelConfig | null = null;
let tokenizer: Map<string, number> | null = null;
let reverseTokenizer: Map<number, string> | null = null;
let isInitialized = false;
let currentProvider: 'cpu' | 'cuda' = 'cpu';

/**
 * Check if HunyuanOCR ONNX models are available
 */
export function isHunyuanAvailable(): boolean {
  // Check for ONNX models
  if (existsSync(join(ONNX_MODEL_DIR, 'vision_encoder.onnx'))) {
    return true;
  }

  // Check for PyTorch models (can be converted)
  if (existsSync(join(PYTORCH_MODEL_DIR, 'config.json'))) {
    return true;
  }

  // Check in default model dir
  if (existsSync(join(MODEL_DIR, 'vision_encoder.onnx'))) {
    return true;
  }

  return false;
}

/**
 * Get the model directory (ONNX or PyTorch)
 */
function getModelDir(): string {
  if (existsSync(join(ONNX_MODEL_DIR, 'vision_encoder.onnx'))) {
    return ONNX_MODEL_DIR;
  }
  if (existsSync(join(MODEL_DIR, 'vision_encoder.onnx'))) {
    return MODEL_DIR;
  }
  return PYTORCH_MODEL_DIR;
}

/**
 * Load preprocessing configuration
 */
function loadPreprocessConfig(modelDir: string): PreprocessConfig {
  const configPath = join(modelDir, 'preprocessor_config.json');

  if (existsSync(configPath)) {
    try {
      const content = readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content);
      return {
        imageSize: config.image_size || DEFAULT_PREPROCESS_CONFIG.imageSize,
        patchSize: config.patch_size || DEFAULT_PREPROCESS_CONFIG.patchSize,
        mean: config.mean || DEFAULT_PREPROCESS_CONFIG.mean,
        std: config.std || DEFAULT_PREPROCESS_CONFIG.std,
        maxImageSize: config.max_image_size || DEFAULT_PREPROCESS_CONFIG.maxImageSize,
      };
    } catch (error) {
      logger.warn('Failed to load preprocessor config, using defaults', { error: String(error) });
    }
  }

  return DEFAULT_PREPROCESS_CONFIG;
}

/**
 * Load model configuration
 */
function loadModelConfig(modelDir: string): ModelConfig | null {
  const configPath = join(modelDir, 'model_config.json');

  if (existsSync(configPath)) {
    try {
      const content = readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content);
      return {
        vision: {
          hiddenSize: config.vision?.hidden_size || 1152,
          numAttentionHeads: config.vision?.num_attention_heads || 16,
          patchSize: config.vision?.patch_size || 16,
          imageSize: config.vision?.image_size || 2048,
        },
        text: {
          hiddenSize: config.text?.hidden_size || 1024,
          vocabSize: config.text?.vocab_size || 120818,
        },
        specialTokens: {
          bosTokenId: config.special_tokens?.bos_token_id || 120000,
          eosTokenId: config.special_tokens?.eos_token_id || 120020,
          padTokenId: config.special_tokens?.pad_token_id || -1,
          imageTokenId: config.special_tokens?.image_token_id || 120120,
        },
      };
    } catch (error) {
      logger.warn('Failed to load model config', { error: String(error) });
    }
  }

  return null;
}

/**
 * Load tokenizer vocabulary
 */
function loadTokenizer(modelDir: string): void {
  const tokenizerPath = join(modelDir, 'tokenizer.json');

  if (!existsSync(tokenizerPath)) {
    logger.warn('Tokenizer not found, text decoding will be limited');
    return;
  }

  try {
    const content = readFileSync(tokenizerPath, 'utf-8');
    const data = JSON.parse(content);

    tokenizer = new Map();
    reverseTokenizer = new Map();

    // Handle different tokenizer formats
    const vocab = data.model?.vocab || data.vocab || {};

    for (const [token, id] of Object.entries(vocab)) {
      const tokenId = typeof id === 'number' ? id : parseInt(id as string, 10);
      tokenizer.set(token, tokenId);
      reverseTokenizer.set(tokenId, token);
    }

    // Add special tokens from added_tokens
    if (data.added_tokens) {
      for (const token of data.added_tokens) {
        if (token.id !== undefined && token.content) {
          tokenizer.set(token.content, token.id);
          reverseTokenizer.set(token.id, token.content);
        }
      }
    }

    logger.info(`Loaded tokenizer with ${tokenizer.size} tokens`);
  } catch (error) {
    logger.error('Failed to load tokenizer', error);
  }
}

/**
 * Create ONNX session with appropriate provider
 */
async function createSession(modelPath: string): Promise<ort.InferenceSession> {
  const hw = detectHardware();
  const useCuda = hw.provider === 'cuda';

  const options: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    logSeverityLevel: 2,
  };

  // Try CUDA first if available
  if (useCuda) {
    try {
      options.executionProviders = [
        { name: 'cuda', deviceId: 0 },
        'cpu',
      ];
      const session = await ort.InferenceSession.create(modelPath, options);
      currentProvider = 'cuda';
      logger.info(`ONNX session created with CUDA provider`);
      return session;
    } catch (error) {
      logger.warn('CUDA provider failed, falling back to CPU', { error: String(error) });
    }
  }

  // Fallback to CPU
  options.executionProviders = ['cpu'];
  const numCpus = require('os').cpus().length;
  options.intraOpNumThreads = Math.max(1, Math.floor(numCpus / 2));
  options.interOpNumThreads = Math.max(1, Math.floor(numCpus / 4));

  const session = await ort.InferenceSession.create(modelPath, options);
  currentProvider = 'cpu';
  logger.info(`ONNX session created with CPU provider (${options.intraOpNumThreads} threads)`);

  return session;
}

/**
 * Initialize HunyuanOCR - load models and tokenizer
 */
export async function initializeHunyuanOCR(): Promise<void> {
  if (isInitialized) {
    return;
  }

  const timer = new Timer('hunyuan-init');
  logger.info('Initializing HunyuanOCR...');

  const modelDir = getModelDir();
  logger.info(`Using model directory: ${modelDir}`);

  // Check for ONNX models
  const visionEncoderPath = join(modelDir, 'vision_encoder.onnx');

  if (!existsSync(visionEncoderPath)) {
    // Try to convert from PyTorch
    logger.info('ONNX model not found, attempting conversion from PyTorch...');
    await convertFromPyTorch();
  }

  // Load configurations
  preprocessConfig = loadPreprocessConfig(modelDir);
  modelConfig = loadModelConfig(modelDir);
  loadTokenizer(modelDir);

  // Load vision encoder
  if (existsSync(visionEncoderPath)) {
    logger.info('Loading vision encoder...');
    visionEncoderSession = await createSession(visionEncoderPath);
    timer.checkpoint('vision-encoder-loaded');
  } else {
    throw new Error('Vision encoder ONNX model not found. Run converter first.');
  }

  // Load text decoder if available
  const textDecoderPath = join(modelDir, 'text_decoder.onnx');
  if (existsSync(textDecoderPath)) {
    logger.info('Loading text decoder...');
    textDecoderSession = await createSession(textDecoderPath);
    timer.checkpoint('text-decoder-loaded');
  }

  isInitialized = true;
  timer.log('HunyuanOCR initialization complete');
}

/**
 * Convert PyTorch model to ONNX (calls Python converter)
 */
async function convertFromPyTorch(): Promise<void> {
  const { execSync } = require('child_process');

  const converterPath = join(__dirname, '../../../converter/convert_hunyuan_ocr.py');

  if (!existsSync(converterPath)) {
    throw new Error('HunyuanOCR converter not found. Please run the converter manually.');
  }

  const hw = detectHardware();
  const cudaFlag = hw.provider === 'cuda' ? '--cuda' : '';

  try {
    logger.info('Running HunyuanOCR ONNX conversion...');
    execSync(
      `python3 ${converterPath} --output ${ONNX_MODEL_DIR} ${cudaFlag}`,
      { stdio: 'inherit', timeout: 600000 } // 10 minute timeout
    );
  } catch (error) {
    logger.error('PyTorch to ONNX conversion failed', error);
    throw new Error('Failed to convert HunyuanOCR to ONNX. Check Python dependencies.');
  }
}

/**
 * Preprocess image for model input
 */
async function preprocessImage(input: Buffer | string): Promise<Float32Array> {
  const timer = new Timer('preprocess-image');

  let imageBuffer: Buffer;

  if (typeof input === 'string') {
    // Check if it's a file path or base64
    if (existsSync(input)) {
      imageBuffer = readFileSync(input);
    } else if (input.startsWith('data:image')) {
      // Base64 data URL
      const base64Data = input.split(',')[1];
      imageBuffer = Buffer.from(base64Data, 'base64');
    } else {
      // Assume raw base64
      imageBuffer = Buffer.from(input, 'base64');
    }
  } else {
    imageBuffer = input;
  }

  // Resize and normalize with sharp
  const { imageSize, mean, std } = preprocessConfig;

  const image = sharp(imageBuffer);
  const metadata = await image.metadata();

  // Calculate resize dimensions (maintain aspect ratio, fit within imageSize)
  let width = metadata.width || imageSize;
  let height = metadata.height || imageSize;

  const maxDim = Math.max(width, height);
  if (maxDim > preprocessConfig.maxImageSize) {
    const scale = preprocessConfig.maxImageSize / maxDim;
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  // Resize to target size
  const resized = await image
    .resize(imageSize, imageSize, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
    .removeAlpha()
    .raw()
    .toBuffer();

  // Convert to float32 and normalize
  const pixels = new Float32Array(3 * imageSize * imageSize);
  const numPixels = imageSize * imageSize;

  for (let i = 0; i < numPixels; i++) {
    const r = resized[i * 3] / 255.0;
    const g = resized[i * 3 + 1] / 255.0;
    const b = resized[i * 3 + 2] / 255.0;

    // Normalize with ImageNet mean/std and convert to CHW format
    pixels[i] = (r - mean[0]) / std[0];                      // R channel
    pixels[numPixels + i] = (g - mean[1]) / std[1];          // G channel
    pixels[2 * numPixels + i] = (b - mean[2]) / std[2];      // B channel
  }

  timer.log('Image preprocessing complete');

  return pixels;
}

/**
 * Run vision encoder inference
 */
async function runVisionEncoder(pixelValues: Float32Array): Promise<Float32Array> {
  if (!visionEncoderSession) {
    throw new Error('Vision encoder not initialized');
  }

  const { imageSize } = preprocessConfig;

  // Create input tensor [batch=1, channels=3, height, width]
  const inputTensor = new ort.Tensor('float32', pixelValues, [1, 3, imageSize, imageSize]);

  // Run inference
  const feeds: Record<string, ort.Tensor> = { pixel_values: inputTensor };
  const results = await visionEncoderSession.run(feeds);

  // Get output
  const outputName = visionEncoderSession.outputNames[0];
  const output = results[outputName];

  return output.data as Float32Array;
}

/**
 * Decode token IDs to text
 */
function decodeTokens(tokenIds: number[]): string {
  if (!reverseTokenizer) {
    return tokenIds.join(' ');
  }

  const tokens: string[] = [];

  for (const id of tokenIds) {
    const token = reverseTokenizer.get(id);
    if (token) {
      // Skip special tokens
      if (!token.startsWith('<') && !token.endsWith('>')) {
        tokens.push(token);
      }
    }
  }

  // Join and clean up
  let text = tokens.join('');

  // Handle BPE-style tokens (spaces encoded as special chars)
  text = text.replace(/\u0120/g, ' ');  // GPT-style space
  text = text.replace(/\u2581/g, ' ');  // SentencePiece style space
  text = text.trim();

  return text;
}

/**
 * Simple greedy decoding for text generation
 */
async function greedyDecode(imageFeatures: Float32Array, maxTokens: number = 512): Promise<number[]> {
  if (!textDecoderSession) {
    // If no decoder, return empty (will use placeholder text)
    logger.warn('Text decoder not available, using vision features only');
    return [];
  }

  const generatedTokens: number[] = [];
  const bosTokenId = modelConfig?.specialTokens.bosTokenId || 120000;
  const eosTokenId = modelConfig?.specialTokens.eosTokenId || 120020;

  // Start with BOS token
  let inputIds = [bosTokenId];

  for (let i = 0; i < maxTokens; i++) {
    // Create input tensors
    const inputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]);
    const attentionMask = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(() => BigInt(1))), [1, inputIds.length]);

    // Run decoder
    const feeds: Record<string, ort.Tensor> = {
      input_ids: inputIdsTensor,
      attention_mask: attentionMask,
    };

    try {
      const results = await textDecoderSession.run(feeds);
      const logits = results[textDecoderSession.outputNames[0]].data as Float32Array;

      // Get next token (greedy - take argmax of last position)
      const vocabSize = modelConfig?.text.vocabSize || 120818;
      const lastPosition = (inputIds.length - 1) * vocabSize;
      let maxLogit = -Infinity;
      let nextToken = eosTokenId;

      for (let j = 0; j < vocabSize; j++) {
        if (logits[lastPosition + j] > maxLogit) {
          maxLogit = logits[lastPosition + j];
          nextToken = j;
        }
      }

      // Check for EOS
      if (nextToken === eosTokenId) {
        break;
      }

      generatedTokens.push(nextToken);
      inputIds.push(nextToken);

    } catch (error) {
      logger.error('Decoder inference failed', error);
      break;
    }
  }

  return generatedTokens;
}

/**
 * Perform OCR on an image
 */
export async function performOCR(
  image: Buffer | string,
  options: HunyuanOCROptions = {}
): Promise<HunyuanOCRResult> {
  const timer = new Timer('hunyuan-ocr');

  // Initialize if needed
  if (!isInitialized) {
    await initializeHunyuanOCR();
  }

  const { maxTokens = 512 } = options;

  // Preprocess image
  const pixelValues = await preprocessImage(image);
  timer.checkpoint('preprocess');

  // Run vision encoder
  const imageFeatures = await runVisionEncoder(pixelValues);
  timer.checkpoint('vision-encoder');

  // Generate text
  let text = '';
  let confidence: number | undefined;

  if (textDecoderSession) {
    const tokenIds = await greedyDecode(imageFeatures, maxTokens);
    text = decodeTokens(tokenIds);
    timer.checkpoint('text-decode');
  } else {
    // Fallback: return placeholder indicating features extracted
    text = `[Vision features extracted: ${imageFeatures.length} values]`;
    logger.warn('Text decoder not available - returning feature summary');
  }

  const processingTime = timer.elapsed();
  timer.log('OCR complete');

  return {
    text,
    confidence,
    processingTime,
    provider: currentProvider,
  };
}

/**
 * Perform batch OCR
 */
export async function performBatchOCR(
  images: Array<Buffer | string>,
  options: HunyuanOCROptions = {}
): Promise<HunyuanOCRResult[]> {
  const results: HunyuanOCRResult[] = [];

  // Process sequentially for now (batch processing would require model changes)
  for (const image of images) {
    const result = await performOCR(image, options);
    results.push(result);
  }

  return results;
}

/**
 * Get current provider info
 */
export function getProviderInfo(): { provider: 'cpu' | 'cuda'; initialized: boolean } {
  return {
    provider: currentProvider,
    initialized: isInitialized,
  };
}

/**
 * Shutdown and release resources
 */
export async function shutdown(): Promise<void> {
  if (visionEncoderSession) {
    await visionEncoderSession.release();
    visionEncoderSession = null;
  }

  if (textDecoderSession) {
    await textDecoderSession.release();
    textDecoderSession = null;
  }

  tokenizer = null;
  reverseTokenizer = null;
  modelConfig = null;
  isInitialized = false;

  logger.info('HunyuanOCR shutdown complete');
}
