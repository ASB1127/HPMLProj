"""
Custom DistilBERT modeling with Randomized SVD (rSVD) for attention layers.
This module defines rSVDLinear, which decomposes a linear layer's weights
using rSVD to reduce parameter count and computational complexity.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification


class rSVDLinear(nn.Module):
    """
    Linear layer decomposed using Randomized SVD.
    Approximates weight matrix W as (A * C) @ B, where A is U_r, C is diagonal S_r, and B is V_r.
    """
    def __init__(self, linear, rank):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        oversample = 8
        in_features = linear.in_features
        out_features = linear.out_features
        
        W = linear.weight.data
        
        q = min(rank + oversample, min(out_features, in_features))

        U, S, V = torch.svd_lowrank(
            W,
            q=q,
            niter=2
        )
        r = min(rank, S.size(0))
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = V[:, :r].T

        self.A = nn.Parameter(U_r.clone())
        self.C = nn.Parameter(S_r.clone())
        self.B = nn.Parameter(V_r.clone())
        
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

        self.rank = r
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        W_approx = (self.A * self.C) @ self.B
        return F.linear(x, W_approx, self.bias)

def apply_rsvd_to_attention_qkv(model, rank):
    """Replaces attention Q, K, V linear layers with rSVDLinear equivalents."""
        for layer in model.distilbert.transformer.layer:
            attn = layer.attention
            attn.q_lin = rSVDLinear(attn.q_lin, rank)
            attn.k_lin = rSVDLinear(attn.k_lin, rank)
            attn.v_lin = rSVDLinear(attn.v_lin, rank)
        return model
class DistilBertForSequenceClassification_rSVD(DistilBertForSequenceClassification):
    """
    DistilBERT model for sequence classification that supports rSVD-decomposed
    attention layers.
    """
    def __init__(self, config):
        super().__init__(config)

        # Extra config flags (saved with the model)
        self.is_rsvd_model = getattr(config, "is_rsvd_model", False)
        self.rsvd_rank = getattr(config, "rsvd_rank", None)

        # If this is an rSVD model, patch the attention layers
        if self.is_rsvd_model and self.rsvd_rank is not None:
            apply_rsvd_to_attention_qkv(self, self.rsvd_rank)


