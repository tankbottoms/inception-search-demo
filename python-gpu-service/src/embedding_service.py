"""
GPU-accelerated Embedding Service using ONNX Runtime
"""
import time
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import onnxruntime as ort
from transformers import AutoTokenizer
import logging

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TimingInfo:
    total_ms: float
    chunking_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None


@dataclass
class ChunkEmbedding:
    chunk_number: int
    chunk: str
    embedding: List[float]


class EmbeddingService:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.initialized = False
        self.model_id: Optional[str] = None
        self.provider: str = "unknown"

    def _get_model_path(self) -> Path:
        """Get path to ONNX model file"""
        model_name = settings.model_name.replace("/", "--")
        model_dir = settings.model_cache_dir / model_name

        # Look for ONNX file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        return onnx_files[0]

    def _create_session_options(self) -> Tuple[ort.SessionOptions, List[str]]:
        """Create ONNX Runtime session options with GPU support"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        providers = []

        if settings.execution_provider == "cuda":
            # Check if CUDA is available
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")

            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": settings.device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB limit
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))
                logger.info(f"Using CUDA provider with device {settings.device_id}")
            else:
                logger.warning("CUDA provider not available, falling back to CPU")

        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")

        return sess_options, providers

    async def initialize(self, model_id: str = "modernbert-embed") -> None:
        """Initialize the embedding service with GPU support"""
        if self.initialized and self.model_id == model_id:
            logger.debug("Embedding service already initialized")
            return

        start_time = time.perf_counter()

        model_path = self._get_model_path()
        tokenizer_path = model_path.parent

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Loading tokenizer from {tokenizer_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Create ONNX session with GPU
        sess_options, providers = self._create_session_options()

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get actual provider being used
        actual_providers = self.session.get_providers()
        self.provider = actual_providers[0] if actual_providers else "unknown"

        self.model_id = model_id
        self.initialized = True

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Embedding service initialized in {elapsed:.2f}ms")
        logger.info(f"Using provider: {self.provider}")
        logger.info(f"Input names: {self.session.get_inputs()}")
        logger.info(f"Output names: {self.session.get_outputs()}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        if not text:
            return ""

        # Normalize unicode
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')

        # Normalize whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('\t', ' ')
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd',
            'corp', 'co', 'no', 'vol', 'rev', 'st', 'ave', 'blvd', 'rd', 'ct',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'u.s', 'u.s.a', 'e.g', 'i.e', 'cf', 'al', 'id', 'op', 'cit',
        }

        normalized = re.sub(r'\s+', ' ', text).strip()
        if not normalized:
            return []

        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', normalized)

        sentences = []
        current = ""

        for part in parts:
            if not part.strip():
                continue

            # Check if previous ended with abbreviation
            last_word = current.strip().split()[-1].lower().rstrip('.') if current.strip() else ""

            if last_word in abbreviations and current:
                current += " " + part
            elif current:
                sentences.append(current.strip())
                current = part
            else:
                current = part

        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _split_into_chunks(self, text: str, prefix: str) -> List[str]:
        """Split text into chunks respecting token limits"""
        processed = self._preprocess_text(text)
        if not processed:
            return []

        sentences = self._sentence_tokenize(processed)
        if not sentences:
            return []

        max_tokens = settings.max_tokens
        overlap_sentences = max(1, int(max_tokens * settings.overlap_ratio))

        chunks = []
        current_chunk: List[str] = []
        current_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=False))

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # If single sentence exceeds limit
            if current_tokens + sentence_tokens > max_tokens - 2:  # Account for special tokens
                if current_chunk:
                    chunks.append(prefix + " ".join(current_chunk))

                # Handle overlap
                overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else []
                current_chunk = overlap + [sentence]
                current_tokens = len(self.tokenizer.encode(prefix + " ".join(current_chunk), add_special_tokens=False))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(prefix + " ".join(current_chunk))

        return chunks

    def _mean_pooling(self, hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Apply mean pooling over hidden states"""
        # hidden_states: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]

        # Expand attention mask for broadcasting
        mask_expanded = np.expand_dims(attention_mask, -1)  # [batch, seq_len, 1]

        # Sum hidden states where mask is 1
        sum_hidden = np.sum(hidden_states * mask_expanded, axis=1)  # [batch, hidden_dim]

        # Count valid tokens
        sum_mask = np.sum(mask_expanded, axis=1)  # [batch, 1]
        sum_mask = np.maximum(sum_mask, 1e-9)  # Avoid division by zero

        # Mean pooling
        return sum_hidden / sum_mask

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return embeddings / norms

    async def _run_inference(self, texts: List[str]) -> np.ndarray:
        """Run inference on batch of texts"""
        if not self.session or not self.tokenizer:
            raise RuntimeError("Service not initialized")

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_tokens,
            return_tensors="np"
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Prepare feeds
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add token_type_ids if required
        input_names = [inp.name for inp in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        # Run inference
        outputs = self.session.run(None, feeds)
        hidden_states = outputs[0]  # [batch, seq_len, hidden_dim]

        # Pool and normalize
        embeddings = self._mean_pooling(hidden_states, attention_mask.astype(np.float32))
        embeddings = self._l2_normalize(embeddings)

        return embeddings

    async def generate_query_embedding(self, text: str) -> Tuple[List[float], TimingInfo]:
        """Generate embedding for a query"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        # Preprocess and add prefix
        processed = self._preprocess_text(text)
        query_text = settings.query_prefix + processed

        preprocess_time = time.perf_counter()

        # Run inference
        embeddings = await self._run_inference([query_text])

        inference_time = time.perf_counter()

        total_ms = (inference_time - start) * 1000
        inference_ms = (inference_time - preprocess_time) * 1000

        return embeddings[0].tolist(), TimingInfo(
            total_ms=total_ms,
            inference_ms=inference_ms
        )

    async def generate_text_embedding(self, doc_id: int, text: str) -> Tuple[List[ChunkEmbedding], TimingInfo]:
        """Generate embeddings for a document with chunking"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        # Split into chunks
        chunks = self._split_into_chunks(text, settings.document_prefix)

        if not chunks:
            raise ValueError("No content to embed after processing")

        chunking_time = time.perf_counter()

        # Process in batches
        all_embeddings = []
        batch_size = settings.processing_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = await self._run_inference(batch)
            all_embeddings.extend(embeddings)

        inference_time = time.perf_counter()

        # Build response
        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            # Remove prefix for response
            chunk_text = chunk.replace(settings.document_prefix, "", 1)
            results.append(ChunkEmbedding(
                chunk_number=i + 1,
                chunk=chunk_text,
                embedding=embedding.tolist()
            ))

        postprocess_time = time.perf_counter()

        return results, TimingInfo(
            total_ms=(postprocess_time - start) * 1000,
            chunking_ms=(chunking_time - start) * 1000,
            inference_ms=(inference_time - chunking_time) * 1000,
            postprocess_ms=(postprocess_time - inference_time) * 1000
        )

    async def generate_batch_embeddings(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], TimingInfo]:
        """Generate embeddings for multiple documents"""
        start = time.perf_counter()

        if not self.initialized:
            await self.initialize()

        # Collect all chunks with metadata
        all_chunks = []
        chunk_meta = []

        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            chunks = self._split_into_chunks(text, settings.document_prefix)

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_meta.append({"doc_id": doc_id, "doc_idx": documents.index(doc)})

        chunking_time = time.perf_counter()

        logger.info(f"Processing batch: {len(documents)} docs, {len(all_chunks)} chunks")

        # Process all chunks
        all_embeddings = []
        batch_size = settings.processing_batch_size

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = await self._run_inference(batch)
            all_embeddings.extend(embeddings)

        inference_time = time.perf_counter()

        # Reconstruct per-document results
        results = {doc["id"]: {"id": doc["id"], "embeddings": []} for doc in documents}

        for i, (chunk, embedding, meta) in enumerate(zip(all_chunks, all_embeddings, chunk_meta)):
            chunk_text = chunk.replace(settings.document_prefix, "", 1)
            doc_result = results[meta["doc_id"]]
            doc_result["embeddings"].append({
                "chunk_number": len(doc_result["embeddings"]) + 1,
                "chunk": chunk_text,
                "embedding": embedding.tolist()
            })

        postprocess_time = time.perf_counter()

        return list(results.values()), TimingInfo(
            total_ms=(postprocess_time - start) * 1000,
            chunking_ms=(chunking_time - start) * 1000,
            inference_ms=(inference_time - chunking_time) * 1000,
            postprocess_ms=(postprocess_time - inference_time) * 1000
        )

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "initialized": self.initialized,
            "model_id": self.model_id,
            "provider": self.provider,
            "embedding_dim": settings.embedding_dim,
        }


# Global instance
embedding_service = EmbeddingService()
