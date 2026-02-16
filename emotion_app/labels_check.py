from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to your fine-tuned text model
MODEL_PATH = './emotion_detection_model'

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Hugging Face models usually store label mapping in config.id2label
id2label = model.config.id2label

print("✅ Model label mapping:")
for idx, label in id2label.items():
    print(f"{idx} -> {label}")
