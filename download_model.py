from transformers import BertTokenizer, BertForSequenceClassification
import os

# SIMPAN DI FOLDER LOKAL DI LUAR ONEDRIVE
SAVE_PATH = r"C:\model_test\indobert_model"

print("⬇️ Mengunduh IndoBERT (format PyTorch, fresh download)...")
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1", force_download=True)
model = BertForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=2,
    ignore_mismatched_sizes=True,
    force_download=True
)

os.makedirs(SAVE_PATH, exist_ok=True)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ Model & tokenizer berhasil disimpan di {SAVE_PATH}")