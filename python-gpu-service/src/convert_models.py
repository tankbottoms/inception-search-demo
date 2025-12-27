"""
Model Converter - Download and convert HuggingFace models to ONNX format
Supports embedding, OCR, and inference models with GPU optimization
"""
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load model registry from JSON file"""
    if not registry_path.exists():
        logger.error(f"Registry file not found: {registry_path}")
        return {"models": []}

    with open(registry_path) as f:
        return json.load(f)


def get_output_dir(model_name: str, cache_dir: Path, suffix: str = "") -> Path:
    """Get output directory for model"""
    safe_name = model_name.replace("/", "--")
    if suffix:
        safe_name = f"{safe_name}-{suffix}"
    return cache_dir / safe_name


def check_torch_available() -> bool:
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def convert_embedding_model_onnx(
    model_id: str,
    model_name: str,
    output_dir: Path,
    optimize_for_gpu: bool = True
) -> bool:
    """Convert embedding model to ONNX format using optimum"""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        logger.info(f"Converting {model_name} to ONNX...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists
        onnx_file = output_dir / "model.onnx"
        if onnx_file.exists():
            logger.info(f"ONNX model already exists at {onnx_file}")
            return True

        # Export to ONNX using optimum
        logger.info(f"Exporting {model_name} to ONNX format...")

        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CUDAExecutionProvider" if optimize_for_gpu else "CPUExecutionProvider"
        )

        # Save model and tokenizer
        model.save_pretrained(str(output_dir))

        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(output_dir))

        logger.info(f"Model saved to {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to convert {model_name} to ONNX: {e}")
        return False


def download_embedding_model_pytorch(
    model_id: str,
    model_name: str,
    output_dir: Path
) -> bool:
    """Download embedding model in PyTorch format using sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Downloading PyTorch model {model_name}...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists
        config_file = output_dir / "config.json"
        if config_file.exists():
            logger.info(f"PyTorch model already exists at {output_dir}")
            return True

        # Download using sentence-transformers
        logger.info(f"Loading model with sentence-transformers...")
        model = SentenceTransformer(model_name)

        # Save to output directory
        model.save(str(output_dir))

        logger.info(f"PyTorch model saved to {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to download PyTorch model {model_name}: {e}")
        return False


def download_ocr_model(
    model_id: str,
    model_name: str,
    output_dir: Path,
    model_config: Dict[str, Any]
) -> bool:
    """Download OCR model - these typically can't be converted to ONNX easily"""
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig

        logger.info(f"Downloading OCR model {model_name}...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists
        config_file = output_dir / "config.json"
        if config_file.exists():
            logger.info(f"OCR model already exists at {output_dir}")
            return True

        # Try to download config first to understand the model
        logger.info(f"Fetching model configuration...")
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.save_pretrained(str(output_dir))
        except Exception as e:
            logger.warning(f"Could not fetch config: {e}")

        # Try to download the processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            processor.save_pretrained(str(output_dir))
            logger.info("Processor saved")
        except Exception as e:
            logger.warning(f"No processor available: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokenizer.save_pretrained(str(output_dir))
                logger.info("Tokenizer saved")
            except Exception as e2:
                logger.warning(f"No tokenizer available: {e2}")

        # Try to download the model
        try:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.save_pretrained(str(output_dir))
            logger.info("Model saved")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    except ImportError as e:
        logger.error(f"Missing required library: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to download OCR model {model_name}: {e}")
        return False


def convert_ocr_model_onnx(
    model_id: str,
    model_name: str,
    output_dir: Path,
    model_config: Dict[str, Any]
) -> bool:
    """Attempt to convert OCR model to ONNX - may not work for all models"""
    logger.warning(f"ONNX conversion for OCR models is experimental")
    logger.info(f"OCR model {model_id} downloaded but not converted to ONNX")
    logger.info("Use the PyTorch model directly for inference")
    return False


def download_inference_model(
    model_id: str,
    model_name: str,
    output_dir: Path,
    model_config: Dict[str, Any]
) -> bool:
    """Download inference/LLM model - typically too large for ONNX conversion"""
    logger.warning(f"Inference model {model_id} is typically too large for ONNX conversion")
    logger.info("Consider using vLLM or TGI for inference model serving")
    logger.info(f"Model: {model_name}")

    # For large inference models, we typically don't download them
    # but provide instructions on how to serve them
    return False


def check_model_status(model_info: Dict[str, Any], cache_dir: Path) -> Dict[str, Any]:
    """Check if model is available and its status"""
    model_id = model_info["id"]
    model_name = model_info["name"]
    model_type = model_info["type"]

    result = {
        "id": model_id,
        "name": model_name,
        "type": model_type,
        "enabled": model_info.get("enabled", False),
        "onnx_available": False,
        "pytorch_available": False,
    }

    # Check ONNX version
    onnx_dir = get_output_dir(model_name, cache_dir)
    if onnx_dir.exists():
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if onnx_files:
            result["onnx_available"] = True
            result["onnx_path"] = str(onnx_dir)
            result["onnx_file"] = str(onnx_files[0])

    # Check PyTorch version
    pytorch_dir = get_output_dir(model_name, cache_dir, "pytorch")
    if pytorch_dir.exists():
        # Check for various indicators of a PyTorch model
        has_pytorch = (
            (pytorch_dir / "config.json").exists() or
            (pytorch_dir / "pytorch_model.bin").exists() or
            (pytorch_dir / "model.safetensors").exists() or
            list(pytorch_dir.glob("*.bin"))
        )
        if has_pytorch:
            result["pytorch_available"] = True
            result["pytorch_path"] = str(pytorch_dir)

    return result


def process_model(
    model_info: Dict[str, Any],
    cache_dir: Path,
    formats: List[str],
    force: bool = False
) -> Dict[str, Any]:
    """Process a single model - download and/or convert"""
    model_id = model_info["id"]
    model_name = model_info["name"]
    model_type = model_info["type"]
    model_config = model_info.get("config", {})

    results = {
        "id": model_id,
        "name": model_name,
        "type": model_type,
        "onnx": {"success": False, "path": None},
        "pytorch": {"success": False, "path": None},
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {model_id} ({model_name})")
    logger.info(f"Type: {model_type}")
    logger.info(f"{'='*60}")

    if model_type == "embedding":
        # Handle embedding models
        if "onnx" in formats:
            output_dir = get_output_dir(model_name, cache_dir)
            success = convert_embedding_model_onnx(model_id, model_name, output_dir)
            results["onnx"]["success"] = success
            results["onnx"]["path"] = str(output_dir) if success else None

        if "pytorch" in formats:
            output_dir = get_output_dir(model_name, cache_dir, "pytorch")
            success = download_embedding_model_pytorch(model_id, model_name, output_dir)
            results["pytorch"]["success"] = success
            results["pytorch"]["path"] = str(output_dir) if success else None

    elif model_type == "ocr":
        # Handle OCR models - download PyTorch by default
        if "pytorch" in formats:
            output_dir = get_output_dir(model_name, cache_dir, "pytorch")
            success = download_ocr_model(model_id, model_name, output_dir, model_config)
            results["pytorch"]["success"] = success
            results["pytorch"]["path"] = str(output_dir) if success else None

        if "onnx" in formats:
            output_dir = get_output_dir(model_name, cache_dir)
            success = convert_ocr_model_onnx(model_id, model_name, output_dir, model_config)
            results["onnx"]["success"] = success
            results["onnx"]["path"] = str(output_dir) if success else None

    elif model_type == "inference":
        # Handle inference models
        output_dir = get_output_dir(model_name, cache_dir)
        success = download_inference_model(model_id, model_name, output_dir, model_config)
        results["pytorch"]["success"] = success
        results["pytorch"]["path"] = str(output_dir) if success else None

    else:
        logger.warning(f"Unknown model type: {model_type}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download and convert models")
    parser.add_argument("--model", type=str, help="Specific model ID to process")
    parser.add_argument("--registry", type=str, default="../models/registry.json",
                       help="Path to model registry JSON")
    parser.add_argument("--cache-dir", type=str, default="../models",
                       help="Model cache directory")
    parser.add_argument("--check", action="store_true", help="Only check model status")
    parser.add_argument("--format", type=str, default="onnx,pytorch",
                       help="Formats to download/convert (comma-separated: onnx,pytorch)")
    parser.add_argument("--type", type=str, help="Only process models of this type")
    parser.add_argument("--enabled-only", action="store_true",
                       help="Only process enabled models")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download/conversion")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    registry_path = Path(args.registry).resolve()
    formats = [f.strip() for f in args.format.split(",")]

    logger.info(f"Model cache directory: {cache_dir}")
    logger.info(f"Registry file: {registry_path}")
    logger.info(f"Formats: {formats}")

    # Load registry
    registry = load_registry(registry_path)
    models = registry.get("models", [])

    if not models:
        logger.error("No models found in registry")
        sys.exit(1)

    # Filter models
    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            logger.error(f"Model not found: {args.model}")
            available = [m["id"] for m in registry.get("models", [])]
            logger.info(f"Available models: {available}")
            sys.exit(1)

    if args.type:
        models = [m for m in models if m["type"] == args.type]

    if args.enabled_only:
        models = [m for m in models if m.get("enabled", False)]

    logger.info(f"Processing {len(models)} model(s)")

    if args.check:
        # Check status of models
        logger.info("\nModel Status:")
        logger.info("-" * 60)
        for model in models:
            status = check_model_status(model, cache_dir)
            enabled = "enabled" if status["enabled"] else "disabled"

            onnx_status = "available" if status["onnx_available"] else "missing"
            pytorch_status = "available" if status["pytorch_available"] else "missing"

            logger.info(f"\n{status['id']} ({status['type']}) [{enabled}]")
            logger.info(f"  Name: {status['name']}")
            logger.info(f"  ONNX:    {onnx_status}")
            if status["onnx_available"]:
                logger.info(f"           {status.get('onnx_path', 'N/A')}")
            logger.info(f"  PyTorch: {pytorch_status}")
            if status["pytorch_available"]:
                logger.info(f"           {status.get('pytorch_path', 'N/A')}")
        return

    # Process models
    all_results = []
    for model in models:
        result = process_model(model, cache_dir, formats, args.force)
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Conversion Summary")
    logger.info("=" * 60)

    for result in all_results:
        onnx_status = "OK" if result["onnx"]["success"] else "--"
        pytorch_status = "OK" if result["pytorch"]["success"] else "--"
        logger.info(f"  {result['id']:20} ONNX: {onnx_status:4} PyTorch: {pytorch_status}")


if __name__ == "__main__":
    main()
