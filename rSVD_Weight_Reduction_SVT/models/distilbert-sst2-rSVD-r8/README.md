# DistilBERT SST-2 — rSVD Low-Rank Model (rank=8)

This model fine-tunes **DistilBERT** on the **SST-2 sentiment classification** task using
**Randomized SVD (rSVD)** to factorize the Q/K/V projection matrices in every transformer
attention layer.

The goal is to measure **memory savings**, **FLOPs reductions**, and **accuracy retention**
as the rank is varied.

---

## 🔧 rSVD Method Summary

- Base model: `distilbert-base-uncased`
- Decomposition applied to: **Q, K, V attention projections**
- rSVD Rank: **8**
- Trainable params: drastically reduced vs full fine-tuning
- Architecture modified using 3-parameter low-rank factors: **A**, **C**, **B**

---

## 📊 Training Configuration

| Setting | Value |
|--------|-------|
| Rank | 8 |
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
| FLOPs per training step | 768,945,524,736 |
| FLOPs per epoch | 1,618,630,329,569,280 |
| Peak GPU Memory (bytes) | 1470103552 |
| Validation Accuracy | 0.8153669724770642 |

*(Metrics are computed automatically during training and profiling.)*

---

## 🚀 Usage Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "AmitBal/distilbert-sst2-rSVD-r8"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```
        