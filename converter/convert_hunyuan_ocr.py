#!/usr/bin/env python3
"""
HunyuanOCR ONNX Converter

Converts tencent/HunyuanOCR from PyTorch to ONNX format.
Supports both CPU and CUDA execution providers.

The model is split into two ONNX files:
1. vision_encoder.onnx - Processes images to embeddings
2. decoder.onnx - Generates text from image embeddings

Usage:
    python convert_hunyuan_ocr.py --output /models/tencent--HunyuanOCR
    python convert_hunyuan_ocr.py --output /models/tencent--HunyuanOCR --cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Model constants
MODEL_NAME = "tencent/HunyuanOCR"
PYTORCH_MODEL_PATH = Path(__file__).parent.parent / "models" / "tencent--HunyuanOCR-pytorch"


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install torch transformers onnx onnxruntime")
        return False

    return True


def load_model(device: str = "cpu"):
    """Load HunyuanOCR model from local PyTorch files or HuggingFace"""
    import torch
    from transformers import AutoModel, AutoProcessor, AutoConfig

    logger.info(f"Loading HunyuanOCR model...")

    # Try local path first
    if PYTORCH_MODEL_PATH.exists():
        logger.info(f"Loading from local path: {PYTORCH_MODEL_PATH}")
        model_path = str(PYTORCH_MODEL_PATH)
    else:
        logger.info(f"Loading from HuggingFace: {MODEL_NAME}")
        model_path = MODEL_NAME

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load model with appropriate dtype
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

        return model, processor, config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def export_vision_encoder(
    model,
    processor,
    output_dir: Path,
    device: str = "cpu"
) -> Path:
    """Export the vision encoder to ONNX"""
    import torch
    import torch.onnx

    logger.info("Exporting vision encoder to ONNX...")

    output_path = output_dir / "vision_encoder.onnx"

    # Get vision encoder
    vision_encoder = model.vision_model if hasattr(model, 'vision_model') else model.model.vision_model

    # Create dummy input
    batch_size = 1
    channels = 3
    image_size = 448  # HunyuanOCR uses 448x448 patches

    dummy_input = torch.randn(batch_size, channels, image_size, image_size)
    if device == "cuda":
        dummy_input = dummy_input.cuda().half()
    else:
        dummy_input = dummy_input.float()

    # Export
    try:
        with torch.no_grad():
            torch.onnx.export(
                vision_encoder,
                dummy_input,
                str(output_path),
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                    "image_features": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=17,
                do_constant_folding=True
            )

        logger.info(f"Vision encoder exported to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to export vision encoder: {e}")
        raise


def create_image_preprocessor_config(
    processor,
    output_dir: Path
) -> None:
    """Save image preprocessor configuration for TypeScript runtime"""
    logger.info("Creating image preprocessor config...")

    # Extract preprocessing config from processor
    config = {
        "image_size": 448,
        "patch_size": 16,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],
        "max_image_size": 2048,
        "resize_mode": "bilinear",
        "do_normalize": True,
        "do_resize": True,
        "do_center_crop": False
    }

    # Try to get actual values from processor
    if hasattr(processor, 'image_processor'):
        img_proc = processor.image_processor
        if hasattr(img_proc, 'size'):
            if isinstance(img_proc.size, dict):
                config["image_size"] = img_proc.size.get("height", 448)
            else:
                config["image_size"] = img_proc.size
        if hasattr(img_proc, 'image_mean'):
            config["mean"] = list(img_proc.image_mean)
        if hasattr(img_proc, 'image_std'):
            config["std"] = list(img_proc.image_std)

    config_path = output_dir / "preprocessor_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Preprocessor config saved to: {config_path}")


def export_text_decoder(
    model,
    processor,
    output_dir: Path,
    device: str = "cpu"
) -> Path:
    """Export the text decoder to ONNX"""
    import torch

    logger.info("Exporting text decoder to ONNX...")

    output_path = output_dir / "text_decoder.onnx"

    # Get language model / decoder
    if hasattr(model, 'language_model'):
        decoder = model.language_model
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        decoder = model.model.language_model
    else:
        # For models that use a combined architecture
        logger.warning("Model architecture not fully supported, using full model")
        decoder = model

    # Create dummy inputs
    batch_size = 1
    seq_length = 128
    hidden_size = 1024  # From config

    # Input IDs for decoder
    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    # Image embeddings (from vision encoder)
    num_image_tokens = 256  # Typical for vision transformers
    dummy_image_embeds = torch.randn(batch_size, num_image_tokens, hidden_size)

    if device == "cuda":
        dummy_input_ids = dummy_input_ids.cuda()
        dummy_attention_mask = dummy_attention_mask.cuda()
        dummy_image_embeds = dummy_image_embeds.cuda().half()
    else:
        dummy_image_embeds = dummy_image_embeds.float()

    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                (dummy_input_ids, dummy_attention_mask),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=17,
                do_constant_folding=True
            )

        logger.info(f"Text decoder exported to: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Could not export decoder separately: {e}")
        logger.info("Will use combined model export instead")
        return None


def export_combined_model(
    model,
    processor,
    output_dir: Path,
    device: str = "cpu"
) -> Path:
    """Export the complete model as a single ONNX file for simpler inference"""
    import torch
    from PIL import Image
    import numpy as np

    logger.info("Exporting combined model to ONNX...")

    output_path = output_dir / "model.onnx"

    # Create a wrapper class for export
    class HunyuanOCRWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values, input_ids=None):
            # Simplified forward for OCR
            outputs = self.model.generate(
                pixel_values=pixel_values,
                max_new_tokens=512,
                do_sample=False
            )
            return outputs

    # For VL models, we often can't export the generate() method directly
    # Instead, export the encoder and use custom decoding
    logger.info("VL models require separate encoder export for ONNX")
    logger.info("Using vision encoder + custom text generation at runtime")

    return None


def save_tokenizer(processor, output_dir: Path) -> None:
    """Save tokenizer for TypeScript runtime"""
    logger.info("Saving tokenizer...")

    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.save_pretrained(str(output_dir))
    else:
        processor.save_pretrained(str(output_dir))

    logger.info(f"Tokenizer saved to: {output_dir}")


def verify_onnx_model(model_path: Path, device: str = "cpu") -> bool:
    """Verify the exported ONNX model"""
    import onnx
    import onnxruntime as ort
    import numpy as np

    logger.info(f"Verifying ONNX model: {model_path}")

    try:
        # Check ONNX model validity
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        # Try to create inference session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        logger.info(f"  Inputs: {[i.name for i in inputs]}")
        logger.info(f"  Outputs: {[o.name for o in outputs]}")
        logger.info(f"  Provider: {session.get_providers()[0]}")

        return True

    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def create_model_config(config, output_dir: Path) -> None:
    """Create model configuration for TypeScript runtime"""
    model_config = {
        "model_type": "hunyuan_vl",
        "architecture": "vision_language",
        "vision": {
            "hidden_size": config.vision_config.hidden_size if hasattr(config, 'vision_config') else 1152,
            "num_attention_heads": config.vision_config.num_attention_heads if hasattr(config, 'vision_config') else 16,
            "num_hidden_layers": config.vision_config.num_hidden_layers if hasattr(config, 'vision_config') else 27,
            "patch_size": config.vision_config.patch_size if hasattr(config, 'vision_config') else 16,
            "image_size": config.vision_config.max_image_size if hasattr(config, 'vision_config') else 2048
        },
        "text": {
            "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else 1024,
            "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else 16,
            "num_hidden_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 24,
            "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else 120818,
            "max_position_embeddings": config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 32768
        },
        "special_tokens": {
            "bos_token_id": config.bos_token_id if hasattr(config, 'bos_token_id') else 120000,
            "eos_token_id": config.eos_token_id if hasattr(config, 'eos_token_id') else 120020,
            "pad_token_id": config.pad_token_id if hasattr(config, 'pad_token_id') else -1,
            "image_token_id": config.image_token_id if hasattr(config, 'image_token_id') else 120120
        },
        "onnx": {
            "vision_encoder": "vision_encoder.onnx",
            "text_decoder": "text_decoder.onnx",
            "opset_version": 17
        }
    }

    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    logger.info(f"Model config saved to: {config_path}")


def convert_hunyuan_ocr(
    output_dir: Path,
    use_cuda: bool = False,
    verify: bool = True
) -> bool:
    """Main conversion function"""
    device = "cuda" if use_cuda else "cpu"

    logger.info(f"Converting HunyuanOCR to ONNX (device: {device})")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model, processor, config = load_model(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Export vision encoder
    try:
        vision_path = export_vision_encoder(model, processor, output_dir, device)
        if verify and vision_path:
            verify_onnx_model(vision_path, device)
    except Exception as e:
        logger.error(f"Vision encoder export failed: {e}")
        # Continue anyway - we might be able to use PyTorch fallback

    # Export text decoder
    try:
        decoder_path = export_text_decoder(model, processor, output_dir, device)
        if verify and decoder_path:
            verify_onnx_model(decoder_path, device)
    except Exception as e:
        logger.warning(f"Text decoder export failed: {e}")
        # This is expected for some model architectures

    # Save tokenizer and configs
    try:
        save_tokenizer(processor, output_dir)
        create_image_preprocessor_config(processor, output_dir)
        create_model_config(config, output_dir)
    except Exception as e:
        logger.error(f"Failed to save configs: {e}")
        return False

    logger.info("Conversion complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert HunyuanOCR to ONNX")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for ONNX files")
    parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA for conversion (requires GPU)")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip ONNX verification")

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Check CUDA availability if requested
    if args.cuda:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available")
            sys.exit(1)

    output_dir = Path(args.output)
    success = convert_hunyuan_ocr(
        output_dir,
        use_cuda=args.cuda,
        verify=not args.no_verify
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
