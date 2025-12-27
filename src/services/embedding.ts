/**
 * Embedding Service - ONNX inference with mean pooling and normalization
 * Mirrors Python EmbeddingService from backup/inception/inception/embedding_service.py
 */

import * as ort from 'onnxruntime-node';
import type { ChunkEmbedding, TextResponse, TimingInfo } from '../types';
import { loadModel, type LoadedModel } from './model-loader';
import { SimpleTokenizer, TextChunker, validateText } from './tokenizer';
import { settings, DOCUMENT_PREFIX, QUERY_PREFIX } from './config';
import { logger, Timer } from './logger';
import { detectHardware } from './hardware';
import { metrics } from './metrics';

// Default model ID for embeddings
const DEFAULT_MODEL_ID = 'modernbert-embed';

// Service state
let currentModel: LoadedModel | null = null;
let tokenizer: SimpleTokenizer | null = null;
let chunker: TextChunker | null = null;
let initialized = false;

/**
 * Initialize the embedding service
 */
export async function initEmbeddingService(modelId = DEFAULT_MODEL_ID): Promise<void> {
  if (initialized && currentModel) {
    logger.debug('Embedding service already initialized');
    return;
  }

  const timer = new Timer('init-embedding-service');
  const hw = detectHardware();

  logger.info(`Initializing embedding service`, {
    modelId,
    provider: hw.provider,
    device: hw.deviceName,
  });

  try {
    // Load the model
    currentModel = await loadModel(modelId);

    // Initialize tokenizer and chunker
    tokenizer = new SimpleTokenizer(currentModel.tokenizer);
    chunker = new TextChunker(tokenizer);

    initialized = true;

    // Record model load time
    const elapsed = timer.elapsed();
    metrics.modelLoadTime.observe(elapsed / 1000);

    logger.info(`Embedding service initialized`, {
      modelId,
      elapsed_ms: elapsed.toFixed(2),
      inputNames: currentModel.session.inputNames,
      outputNames: currentModel.session.outputNames,
    });
  } catch (error) {
    logger.error('Failed to initialize embedding service', error);
    throw error;
  }
}

/**
 * Ensure service is initialized
 */
async function ensureInitialized(): Promise<void> {
  if (!initialized || !currentModel || !tokenizer || !chunker) {
    await initEmbeddingService();
  }
}

/**
 * Prepare input tensors for the model
 */
function prepareInputs(
  texts: string[],
  maxLength: number
): { inputIds: ort.Tensor; attentionMask: ort.Tensor } {
  if (!tokenizer) throw new Error('Tokenizer not initialized');

  const batchSize = texts.length;

  // Tokenize all texts
  const tokenizedTexts = texts.map(text => tokenizer!.encode(text, true));

  // Find max length in batch (capped by model max)
  const actualMaxLength = Math.min(
    Math.max(...tokenizedTexts.map(t => t.length)),
    maxLength
  );

  // Create padded arrays
  const inputIds = new BigInt64Array(batchSize * actualMaxLength);
  const attentionMask = new BigInt64Array(batchSize * actualMaxLength);

  for (let i = 0; i < batchSize; i++) {
    const tokens = tokenizedTexts[i];
    const offset = i * actualMaxLength;

    for (let j = 0; j < actualMaxLength; j++) {
      if (j < tokens.length) {
        inputIds[offset + j] = BigInt(tokens[j]);
        attentionMask[offset + j] = 1n;
      } else {
        inputIds[offset + j] = BigInt(tokenizer!.padId);
        attentionMask[offset + j] = 0n;
      }
    }
  }

  return {
    inputIds: new ort.Tensor('int64', inputIds, [batchSize, actualMaxLength]),
    attentionMask: new ort.Tensor('int64', attentionMask, [batchSize, actualMaxLength]),
  };
}

/**
 * Mean pooling over hidden states with attention mask
 */
function meanPooling(
  hiddenStates: Float32Array,
  attentionMask: BigInt64Array,
  batchSize: number,
  seqLength: number,
  hiddenDim: number
): Float32Array {
  const embeddings = new Float32Array(batchSize * hiddenDim);

  for (let b = 0; b < batchSize; b++) {
    const batchOffset = b * seqLength * hiddenDim;
    const maskOffset = b * seqLength;
    let validTokens = 0;

    // Sum hidden states for valid tokens
    for (let s = 0; s < seqLength; s++) {
      if (attentionMask[maskOffset + s] === 1n) {
        validTokens++;
        for (let d = 0; d < hiddenDim; d++) {
          embeddings[b * hiddenDim + d] += hiddenStates[batchOffset + s * hiddenDim + d];
        }
      }
    }

    // Average by valid token count
    if (validTokens > 0) {
      for (let d = 0; d < hiddenDim; d++) {
        embeddings[b * hiddenDim + d] /= validTokens;
      }
    }
  }

  return embeddings;
}

/**
 * L2 normalize embeddings
 */
function l2Normalize(embeddings: Float32Array, batchSize: number, hiddenDim: number): Float32Array {
  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenDim;
    let norm = 0;

    // Calculate L2 norm
    for (let d = 0; d < hiddenDim; d++) {
      norm += embeddings[offset + d] ** 2;
    }
    norm = Math.sqrt(norm);

    // Normalize
    if (norm > 0) {
      for (let d = 0; d < hiddenDim; d++) {
        embeddings[offset + d] /= norm;
      }
    }
  }

  return embeddings;
}

/**
 * Run inference on a batch of texts
 */
async function runInference(texts: string[]): Promise<number[][]> {
  if (!currentModel) throw new Error('Model not loaded');

  const timer = new Timer('inference');
  const batchSize = texts.length;
  const maxLength = currentModel.config.config?.max_tokens || settings.maxTokens;
  const hiddenDim = currentModel.config.config?.embedding_dim || 768;

  // Prepare inputs
  const { inputIds, attentionMask } = prepareInputs(texts, maxLength);
  timer.checkpoint('prepare-inputs');

  // Determine input names (models may use different names)
  const session = currentModel.session;
  const inputNames = session.inputNames;

  const feeds: Record<string, ort.Tensor> = {};

  // Map to actual input names
  if (inputNames.includes('input_ids')) {
    feeds['input_ids'] = inputIds;
  } else if (inputNames.includes('inputs')) {
    feeds['inputs'] = inputIds;
  }

  if (inputNames.includes('attention_mask')) {
    feeds['attention_mask'] = attentionMask;
  }

  // Some models require token_type_ids
  if (inputNames.includes('token_type_ids')) {
    const seqLength = inputIds.dims[1] as number;
    const tokenTypeIds = new BigInt64Array(batchSize * seqLength).fill(0n);
    feeds['token_type_ids'] = new ort.Tensor('int64', tokenTypeIds, [batchSize, seqLength]);
  }

  // Run inference
  const results = await session.run(feeds);
  timer.checkpoint('run-model');

  // Get output tensor
  const outputNames = session.outputNames;
  const outputName = outputNames.find(n =>
    n.includes('last_hidden_state') || n.includes('output') || n === 'sentence_embedding'
  ) || outputNames[0];

  const output = results[outputName];

  if (!output) {
    throw new Error(`No output found. Available: ${outputNames.join(', ')}`);
  }

  const outputData = output.data as Float32Array;
  const outputDims = output.dims as number[];

  // Handle different output formats
  let embeddings: Float32Array;
  const seqLength = inputIds.dims[1] as number;

  if (outputDims.length === 3) {
    // Output is [batch, seq, hidden] - need pooling
    embeddings = meanPooling(
      outputData,
      attentionMask.data as BigInt64Array,
      batchSize,
      seqLength,
      hiddenDim
    );
  } else if (outputDims.length === 2) {
    // Output is already [batch, hidden] - sentence embedding
    embeddings = outputData;
  } else {
    throw new Error(`Unexpected output dimensions: ${outputDims.join(', ')}`);
  }

  timer.checkpoint('pooling');

  // Normalize if configured
  if (currentModel.config.config?.normalize !== false) {
    embeddings = l2Normalize(embeddings, batchSize, hiddenDim);
  }

  timer.checkpoint('normalize');

  // Convert to array of arrays
  const result: number[][] = [];
  for (let b = 0; b < batchSize; b++) {
    const embedding = Array.from(embeddings.slice(b * hiddenDim, (b + 1) * hiddenDim));
    result.push(embedding);
  }

  logger.debug(`Inference complete`, {
    batchSize,
    elapsed_ms: timer.elapsed().toFixed(2),
    checkpoints: timer.getCheckpoints(),
  });

  return result;
}

/**
 * Generate embedding for a query
 */
export async function generateQueryEmbedding(text: string): Promise<{
  embedding: number[];
  timing: TimingInfo;
}> {
  await ensureInitialized();
  const timer = new Timer('query-embedding');

  // Validate input
  const validation = validateText(text, true);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  // Prepare query text
  const queryText = chunker!.processQuery(text);
  timer.checkpoint('preprocess');

  // Run inference
  const embeddings = await runInference([queryText]);
  timer.checkpoint('inference');

  const elapsed = timer.elapsed();
  metrics.processingTime.observe({ endpoint: 'query' }, elapsed / 1000);
  metrics.requestCount.inc({ endpoint: 'query', status: 'success' });

  return {
    embedding: embeddings[0],
    timing: {
      total_ms: elapsed,
      inference_ms: timer.getCheckpoints()['inference'] - timer.getCheckpoints()['preprocess'],
    },
  };
}

/**
 * Generate embeddings for a single document
 */
export async function generateTextEmbedding(
  id: number,
  text: string
): Promise<{ response: TextResponse; timing: TimingInfo }> {
  await ensureInitialized();
  const timer = new Timer('text-embedding');

  // Validate input
  const validation = validateText(text, false);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  // Chunk the text
  const chunks = chunker!.splitIntoChunks(text, DOCUMENT_PREFIX);
  timer.checkpoint('chunking');

  if (chunks.length === 0) {
    throw new Error('No content to embed after processing');
  }

  metrics.chunkCount.inc({ endpoint: 'text' }, chunks.length);

  // Run inference on all chunks
  const embeddings = await runInference(chunks);
  timer.checkpoint('inference');

  // Build response
  const chunkEmbeddings: ChunkEmbedding[] = chunks.map((chunk, idx) => ({
    chunk_number: idx + 1,
    chunk: chunk.replace(DOCUMENT_PREFIX, ''), // Remove prefix for response
    embedding: embeddings[idx],
  }));

  timer.checkpoint('postprocess');
  const elapsed = timer.elapsed();

  metrics.processingTime.observe({ endpoint: 'text' }, elapsed / 1000);
  metrics.requestCount.inc({ endpoint: 'text', status: 'success' });

  return {
    response: {
      id,
      embeddings: chunkEmbeddings,
    },
    timing: {
      total_ms: elapsed,
      chunking_ms: timer.getCheckpoints()['chunking'],
      inference_ms: timer.getCheckpoints()['inference'] - timer.getCheckpoints()['chunking'],
      postprocess_ms: timer.getCheckpoints()['postprocess'] - timer.getCheckpoints()['inference'],
    },
  };
}

/**
 * Generate embeddings for a batch of documents
 */
export async function generateBatchEmbeddings(
  documents: Array<{ id: number; text: string }>
): Promise<{ results: TextResponse[]; timing: TimingInfo }> {
  await ensureInitialized();
  const timer = new Timer('batch-embedding');

  if (documents.length === 0) {
    throw new Error('No documents provided');
  }

  if (documents.length > settings.maxBatchSize) {
    throw new Error(`Batch size ${documents.length} exceeds maximum of ${settings.maxBatchSize}`);
  }

  logger.info(`Processing batch of ${documents.length} documents`);

  // Validate all documents
  for (const doc of documents) {
    const validation = validateText(doc.text, false);
    if (!validation.valid) {
      throw new Error(`Document ${doc.id}: ${validation.error}`);
    }
  }

  // Chunk all documents
  const allChunks: string[] = [];
  const chunkMeta: Array<{ docIdx: number; docId: number }> = [];

  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    const chunks = chunker!.splitIntoChunks(doc.text, DOCUMENT_PREFIX);

    for (const chunk of chunks) {
      allChunks.push(chunk);
      chunkMeta.push({ docIdx: i, docId: doc.id });
    }

    metrics.chunkCount.inc({ endpoint: 'batch' }, chunks.length);
  }

  timer.checkpoint('chunking');
  logger.info(`Total chunks to process: ${allChunks.length}`);

  // Process in batches
  const batchSize = settings.processingBatchSize;
  const allEmbeddings: number[][] = [];

  for (let i = 0; i < allChunks.length; i += batchSize) {
    const batchChunks = allChunks.slice(i, i + batchSize);
    const batchEmbeddings = await runInference(batchChunks);
    allEmbeddings.push(...batchEmbeddings);
  }

  timer.checkpoint('inference');

  // Reconstruct results per document
  const results: TextResponse[] = documents.map(doc => ({
    id: doc.id,
    embeddings: [],
  }));

  for (let i = 0; i < allChunks.length; i++) {
    const { docIdx, docId } = chunkMeta[i];
    const chunk = allChunks[i].replace(DOCUMENT_PREFIX, '');
    const embedding = allEmbeddings[i];

    results[docIdx].embeddings.push({
      chunk_number: results[docIdx].embeddings.length + 1,
      chunk,
      embedding,
    });
  }

  timer.checkpoint('postprocess');
  const elapsed = timer.elapsed();

  metrics.processingTime.observe({ endpoint: 'batch' }, elapsed / 1000);
  metrics.requestCount.inc({ endpoint: 'batch', status: 'success' });

  logger.info(`Batch processing complete`, {
    documents: documents.length,
    chunks: allChunks.length,
    elapsed_ms: elapsed.toFixed(2),
  });

  return {
    results,
    timing: {
      total_ms: elapsed,
      chunking_ms: timer.getCheckpoints()['chunking'],
      inference_ms: timer.getCheckpoints()['inference'] - timer.getCheckpoints()['chunking'],
      postprocess_ms: timer.getCheckpoints()['postprocess'] - timer.getCheckpoints()['inference'],
    },
  };
}

/**
 * Get service status
 */
export function getServiceStatus(): {
  initialized: boolean;
  modelId?: string;
  provider?: string;
  embeddingDim?: number;
} {
  const hw = detectHardware();

  return {
    initialized,
    modelId: currentModel?.config.id,
    provider: hw.provider,
    embeddingDim: currentModel?.config.config?.embedding_dim,
  };
}

/**
 * Shutdown the service
 */
export async function shutdownEmbeddingService(): Promise<void> {
  if (currentModel) {
    currentModel.session.release();
    currentModel = null;
  }
  tokenizer = null;
  chunker = null;
  initialized = false;
  logger.info('Embedding service shut down');
}
