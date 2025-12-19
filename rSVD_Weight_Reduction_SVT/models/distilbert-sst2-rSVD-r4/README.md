# DistilBERT SST-2 — rSVD Low-Rank Model (rank=4)

This model fine-tunes **DistilBERT** on the **SST-2 sentiment classification** task using
**Randomized SVD (rSVD)** to factorize the Q/K/V projection matrices in every transformer
attention layer.

The goal is to measure **memory savings**, **FLOPs reductions**, and **accuracy retention**
as the rank is varied.

---

## 🔧 rSVD Method Summary

- Base model: `distilbert-base-uncased`
- Decomposition applied to: **Q, K, V attention projections**
- rSVD Rank: **4**
- Trainable params: drastically reduced vs full fine-tuning
- Architecture modified using 3-parameter low-rank factors: **A**, **C**, **B**

---

## 📊 Training Configuration

| Setting | Value |
|--------|-------|
| Rank | 4 |
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
| FLOPs per training step | 768,690,554,880 |
| FLOPs per epoch | 1,618,093,618,022,400 |
| Peak GPU Memory (bytes) | 1480589312 |
| Validation Accuracy | 0.8279816513761468 |

*(Metrics are computed automatically during training and profiling.)*

---

## 🚀 Usage Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "AmitBal/distilbert-sst2-rSVD-r4"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```
        