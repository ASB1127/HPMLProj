import torch
from transformers import AutoTokenizer
from rSVD_Weight_Reduction.rSVD_HF_config.rSVD_modeling_distilbert import (
    DistilBertForSequenceClassification_rSVD
)

import sys
from pathlib import Path
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def print_model(model_dir):
    model_dir = Path(model_dir)

    print(f"\n=== Loading model from {model_dir} ===")

    # Load model + tokenizer
    model = DistilBertForSequenceClassification_rSVD.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("\n=== MODEL ARCHITECTURE ===")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n=== TRAINABLE PARAMETER LIST ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:60} {tuple(p.shape)}")

    # Print rSVD-modified Q/K/V layers
    print("\n=== Q/K/V rSVD Layers (per Transformer layer) ===")
    for i, layer in enumerate(model.distilbert.transformer.layer):
        print(f"\nLayer {i}:")
        print("  Q_lin:", layer.attention.q_lin)
        print("  K_lin:", layer.attention.k_lin)
        print("  V_lin:", layer.attention.v_lin)

    print("\n=== DONE ===")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_model.py <model_directory>")
        sys.exit(1)

    model_dir = sys.argv[1]
    print_model(model_dir)

