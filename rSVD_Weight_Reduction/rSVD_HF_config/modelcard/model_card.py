class ModelCard:
    def write_model_card(
        self,
        path,
        rank,
        flops_step=None,
        flops_epoch=None,
        peak_memory=None,
        val_accuracy=None,
        learning_rate=None,
        num_epochs=None,
    ):
        """
        Generates a README.md model card for HuggingFace.
        """

        readme_path = Path(path) / "README.md"

        content = f"""# DistilBERT SST-2 — rSVD Low-Rank Model (rank={rank})

This model fine-tunes **DistilBERT** on the **SST-2 sentiment classification** task using
**Randomized SVD (rSVD)** to factorize the Q/K/V projection matrices in every transformer
attention layer.

The goal is to measure **memory savings**, **FLOPs reductions**, and **accuracy retention**
as the rank is varied.

---

## 🔧 rSVD Method Summary

- Base model: `distilbert-base-uncased`
- Decomposition applied to: **Q, K, V attention projections**
- rSVD Rank: **{rank}**
- Trainable params: drastically reduced vs full fine-tuning
- Architecture modified using 3-parameter low-rank factors: **A**, **C**, **B**

---

## 📊 Training Configuration

| Setting | Value |
|--------|-------|
| Rank | {rank} |
| Learning Rate | {learning_rate} |
| Num Epochs | {num_epochs} |
| Batch Size | 32 |
| Dataset | GLUE SST-2 |
| Optimizer | AdamW |
| Scheduler | Constant LR |

---

## 🧮 Performance Metrics

| Metric | Value |
|--------|--------|
| FLOPs per training step | {flops_step:,} |
| FLOPs per epoch | {flops_epoch:,} |
| Peak GPU Memory (bytes) | {peak_memory} |
| Validation Accuracy | {val_accuracy} |

*(Metrics are computed automatically during training and profiling.)*

---

## 🚀 Usage Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "AmitBal/distilbert-sst2-rSVD-r{rank}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```
        """

        readme_path.write_text(content)