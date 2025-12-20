"""
Script to download and save the base DistilBERT model and tokenizer locally.
This ensures the base model is available for weight merging and offline usage.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
).save_pretrained("/workspace/HPMLProj/Lora/distilbert-original")

AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
).save_pretrained("/workspace/HPMLProj/Lora/distilbert-original")

