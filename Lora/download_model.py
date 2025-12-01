from transformers import AutoModelForSequenceClassification, AutoTokenizer

AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
).save_pretrained("/workspace/HPMLProj/Lora/distilbert-original")

AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
).save_pretrained("/workspace/HPMLProj/Lora/distilbert-original")

