#!/usr/bin/env python3
"""
ONNX Model Converter CLI

Converts HuggingFace models to ONNX format.

Usage:
    python convert.py <model_name> [--output <path>]
    python convert.py --from-registry <registry.json>
"""

import argparse
import json
import os
import sys
from pathlib import Path


def convert_model(model_name: str, output_dir: str) -> dict:
    """Convert a HuggingFace model to ONNX format."""
    print(f"[INFO] Converting model: {model_name}")
    print(f"[INFO] Output directory: {output_dir}")

    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoTokenizer

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Export to ONNX
        print("[INFO] Exporting to ONNX...")
        main_export(
            model_name,
            output_dir,
            task="feature-extraction",
            opset=14,
        )

        # Save tokenizer
        print("[INFO] Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # Extract and save pooling config
        print("[INFO] Extracting pooling config...")
        pooling_config = {
            "mode": "mean",
            "normalize": True
        }
        with open(os.path.join(output_dir, "pooling_config.json"), "w") as f:
            json.dump(pooling_config, f, indent=2)

        print("[OK] Conversion complete")
        return {"status": "success", "output": output_dir}

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Converter")
    parser.add_argument("model", nargs="?", help="Model name or HuggingFace ID")
    parser.add_argument("--output", "-o", default="/models", help="Output directory")
    parser.add_argument("--from-registry", help="Convert all models from registry JSON")

    args = parser.parse_args()

    if args.from_registry:
        # Convert all models from registry
        with open(args.from_registry) as f:
            registry = json.load(f)

        for model in registry.get("models", []):
            if model.get("enabled") and model.get("type") == "embedding":
                model_name = model["name"]
                output_dir = os.path.join(args.output, model["id"])
                convert_model(model_name, output_dir)
    elif args.model:
        # Convert single model
        model_id = args.model.split("/")[-1]
        output_dir = os.path.join(args.output, model_id)
        convert_model(args.model, output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
