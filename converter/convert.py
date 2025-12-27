#!/usr/bin/env python3
"""
ONNX Model Converter CLI

Converts HuggingFace models to ONNX format.
Supports embedding models and OCR models.

Usage:
    python convert.py <model_name> [--output <path>]
    python convert.py --from-registry <registry.json>
    python convert.py --auto  # Auto-convert from /models/registry.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def is_model_converted(model_id: str, models_dir: str) -> bool:
    """Check if a model has already been converted to ONNX"""
    model_path = Path(models_dir) / model_id

    # Check for ONNX model file
    onnx_files = list(model_path.glob("*.onnx"))
    if onnx_files:
        logger.info(f"[SKIP] Model {model_id} already converted: {onnx_files[0].name}")
        return True

    # Also check for model.onnx in subdirectory patterns
    for pattern in ["model.onnx", "vision_encoder.onnx", "onnx/model.onnx"]:
        if (model_path / pattern).exists():
            logger.info(f"[SKIP] Model {model_id} already converted: {pattern}")
            return True

    return False


def convert_embedding_model(model_name: str, output_dir: str) -> dict:
    """Convert a HuggingFace embedding model to ONNX format."""
    logger.info(f"[EMBEDDING] Converting: {model_name}")
    logger.info(f"[EMBEDDING] Output: {output_dir}")

    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Export to ONNX
        logger.info("[EMBEDDING] Exporting to ONNX...")
        main_export(
            model_name,
            output_dir,
            task="feature-extraction",
            opset=14,
        )

        # Save tokenizer
        logger.info("[EMBEDDING] Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # Extract and save pooling config
        logger.info("[EMBEDDING] Saving pooling config...")
        pooling_config = {
            "mode": "mean",
            "normalize": True
        }
        with open(os.path.join(output_dir, "pooling_config.json"), "w") as f:
            json.dump(pooling_config, f, indent=2)

        logger.info("[EMBEDDING] Conversion complete")
        return {"status": "success", "output": output_dir}

    except Exception as e:
        logger.error(f"[EMBEDDING] Conversion failed: {e}")
        return {"status": "error", "error": str(e)}


def convert_ocr_model(model_name: str, output_dir: str, use_cuda: bool = False) -> dict:
    """Convert an OCR model to ONNX format."""
    logger.info(f"[OCR] Converting: {model_name}")
    logger.info(f"[OCR] Output: {output_dir}")

    # Check if it's HunyuanOCR
    if "hunyuan" in model_name.lower():
        return convert_hunyuan_ocr(model_name, output_dir, use_cuda)

    # Generic OCR conversion (fallback)
    logger.warning(f"[OCR] No specific converter for {model_name}, trying generic export")
    return convert_embedding_model(model_name, output_dir)


def convert_hunyuan_ocr(model_name: str, output_dir: str, use_cuda: bool = False) -> dict:
    """Convert HunyuanOCR to ONNX format."""
    import torch

    logger.info("[OCR:HunyuanOCR] Starting conversion...")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"[OCR:HunyuanOCR] Device: {device}")

    # Check for local PyTorch model first
    models_base = Path(os.environ.get("MODELS_DIR", "/models"))
    pytorch_path = models_base / "tencent--HunyuanOCR-pytorch"

    if pytorch_path.exists():
        model_path = str(pytorch_path)
        logger.info(f"[OCR:HunyuanOCR] Using local model: {model_path}")
    else:
        model_path = model_name
        logger.info(f"[OCR:HunyuanOCR] Downloading from HuggingFace: {model_path}")

    try:
        from transformers import AutoModel, AutoProcessor, AutoConfig

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load model
        logger.info("[OCR:HunyuanOCR] Loading model...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None
        )

        if device == "cpu":
            model = model.to(device)
        model.eval()

        # Export vision encoder
        logger.info("[OCR:HunyuanOCR] Exporting vision encoder...")
        vision_encoder = model.vision_model if hasattr(model, 'vision_model') else model.model.vision_model

        batch_size = 1
        channels = 3
        image_size = 448

        dummy_input = torch.randn(batch_size, channels, image_size, image_size)
        if device == "cuda":
            dummy_input = dummy_input.cuda().half()
        else:
            dummy_input = dummy_input.float()

        vision_path = os.path.join(output_dir, "vision_encoder.onnx")

        with torch.no_grad():
            torch.onnx.export(
                vision_encoder,
                dummy_input,
                vision_path,
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                    "image_features": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=17,
                do_constant_folding=True
            )

        logger.info(f"[OCR:HunyuanOCR] Vision encoder saved: {vision_path}")

        # Save tokenizer
        logger.info("[OCR:HunyuanOCR] Saving tokenizer...")
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.save_pretrained(output_dir)
        else:
            processor.save_pretrained(output_dir)

        # Save preprocessor config
        logger.info("[OCR:HunyuanOCR] Saving preprocessor config...")
        preprocessor_config = {
            "image_size": 448,
            "patch_size": 16,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "max_image_size": 2048,
            "resize_mode": "bilinear",
            "do_normalize": True,
            "do_resize": True
        }

        with open(os.path.join(output_dir, "preprocessor_config.json"), "w") as f:
            json.dump(preprocessor_config, f, indent=2)

        # Save model config
        logger.info("[OCR:HunyuanOCR] Saving model config...")
        model_config = {
            "model_type": "hunyuan_vl",
            "architecture": "vision_language",
            "onnx": {
                "vision_encoder": "vision_encoder.onnx",
                "opset_version": 17
            }
        }

        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        logger.info("[OCR:HunyuanOCR] Conversion complete")
        return {"status": "success", "output": output_dir}

    except Exception as e:
        logger.error(f"[OCR:HunyuanOCR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def convert_model(model_name: str, output_dir: str, model_type: str = "embedding") -> dict:
    """Convert a model to ONNX format based on its type."""
    if model_type == "ocr":
        return convert_ocr_model(model_name, output_dir)
    else:
        return convert_embedding_model(model_name, output_dir)


def convert_from_registry(registry_path: str, models_dir: str, force: bool = False) -> dict:
    """
    Convert all enabled models from a registry file.

    Args:
        registry_path: Path to the registry.json file
        models_dir: Base directory for model storage
        force: If True, re-convert even if model exists

    Returns:
        Summary of conversion results
    """
    logger.info(f"[REGISTRY] Loading: {registry_path}")
    logger.info(f"[REGISTRY] Models dir: {models_dir}")

    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"[REGISTRY] Failed to load registry: {e}")
        return {"status": "error", "error": str(e)}

    results = {
        "converted": [],
        "skipped": [],
        "failed": []
    }

    models = registry.get("models", [])
    logger.info(f"[REGISTRY] Found {len(models)} models in registry")

    for model in models:
        model_id = model.get("id")
        model_name = model.get("name")
        model_type = model.get("type", "embedding")
        enabled = model.get("enabled", False)
        formats = model.get("formats", [])

        logger.info(f"\n[MODEL] {model_id} ({model_type})")
        logger.info(f"  Name: {model_name}")
        logger.info(f"  Enabled: {enabled}")
        logger.info(f"  Formats: {formats}")

        # Skip disabled models
        if not enabled:
            logger.info(f"  [SKIP] Model is disabled")
            results["skipped"].append({"id": model_id, "reason": "disabled"})
            continue

        # Skip models that don't support ONNX
        if "onnx" not in formats:
            logger.info(f"  [SKIP] Model does not support ONNX format")
            results["skipped"].append({"id": model_id, "reason": "no_onnx_support"})
            continue

        # Skip inference models (like LLMs) - they need special handling
        if model_type == "inference":
            logger.info(f"  [SKIP] Inference models require vLLM/TensorRT")
            results["skipped"].append({"id": model_id, "reason": "inference_model"})
            continue

        # Determine output directory
        # Use HuggingFace-style naming: org--model-name
        safe_name = model_name.replace("/", "--")
        output_dir = os.path.join(models_dir, safe_name)

        # Check if already converted
        if not force and is_model_converted(safe_name, models_dir):
            results["skipped"].append({"id": model_id, "reason": "already_converted"})
            continue

        # Convert based on model type
        logger.info(f"  [CONVERT] Starting conversion...")

        if model_type == "embedding":
            result = convert_embedding_model(model_name, output_dir)
        elif model_type == "ocr":
            result = convert_ocr_model(model_name, output_dir)
        else:
            logger.warning(f"  [SKIP] Unknown model type: {model_type}")
            results["skipped"].append({"id": model_id, "reason": f"unknown_type_{model_type}"})
            continue

        if result["status"] == "success":
            logger.info(f"  [OK] Conversion successful")
            results["converted"].append({"id": model_id, "output": output_dir})
        else:
            logger.error(f"  [FAIL] Conversion failed: {result.get('error')}")
            results["failed"].append({"id": model_id, "error": result.get("error")})

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("[SUMMARY] Conversion Results")
    logger.info("=" * 60)
    logger.info(f"  Converted: {len(results['converted'])}")
    logger.info(f"  Skipped:   {len(results['skipped'])}")
    logger.info(f"  Failed:    {len(results['failed'])}")

    if results["converted"]:
        logger.info("\n  Converted models:")
        for m in results["converted"]:
            logger.info(f"    - {m['id']}")

    if results["failed"]:
        logger.info("\n  Failed models:")
        for m in results["failed"]:
            logger.info(f"    - {m['id']}: {m['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Converter")
    parser.add_argument("model", nargs="?", help="Model name or HuggingFace ID")
    parser.add_argument("--output", "-o", default="/models", help="Output directory")
    parser.add_argument("--from-registry", help="Convert all models from registry JSON")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-convert from /models/registry.json")
    parser.add_argument("--type", "-t", default="embedding",
                        choices=["embedding", "ocr"],
                        help="Model type (embedding or ocr)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-conversion even if model exists")

    args = parser.parse_args()

    if args.auto:
        # Auto-convert from default registry location
        registry_path = os.path.join(args.output, "registry.json")
        if not os.path.exists(registry_path):
            logger.error(f"Registry not found: {registry_path}")
            sys.exit(1)
        convert_from_registry(registry_path, args.output, force=args.force)
    elif args.from_registry:
        # Convert all models from registry
        convert_from_registry(args.from_registry, args.output, force=args.force)
    elif args.model:
        # Convert single model
        model_id = args.model.split("/")[-1]
        output_dir = os.path.join(args.output, model_id)
        result = convert_model(args.model, output_dir, model_type=args.type)
        if result["status"] == "error":
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
