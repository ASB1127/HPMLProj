# DistilBERT SST-2 — rSVD Low-Rank Model (rank=16)

This model fine-tunes **DistilBERT** on the **SST-2 sentiment classification** task using
**Randomized SVD (rSVD)** to factorize the Q/K/V projection matrices in every transformer
attention layer.

The goal is to measure **memory savings**, **FLOPs reductions**, and **accuracy retention**
as the rank is varied.

---

## 🔧 rSVD Method Summary

- Base model: `distilbert-base-uncased`
- Decomposition applied to: **Q, K, V attention projections**
- rSVD Rank: **16**
- Trainable params: drastically reduced vs full fine-tuning
- Architecture modified using 3-parameter low-rank factors: **A**, **C**, **B**

---

## 📊 Training Configuration

| Setting | Value |
|--------|-------|
| Rank | 16 |
| Learning Rate | 0.0002 |
| Num Epochs | 10 |
| Batch Size | 32 |
| Dataset | GLUE SST-2 |
| Optimizer | AdamW |
| Scheduler | Constant LR |

---

## 🧮 Performance Metrics

| Metric | Value |
|--------|--------|
| FLOPs per training step | 769,455,464,448 |
| FLOPs per epoch | 601,714,173,198,336 |
| Peak GPU Memory (bytes) | 1478492160 |
| Validation Accuracy | 0.81476 |

*(Metrics are computed automatically during training and profiling.)*

---

## 🚀 Usage Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "AmitBal/distilbert-sst2-rSVD-r16"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```
        