from datasets import load_dataset, DatasetDict
import os

# -------------------- Step 1: Load dataset --------------------
print("Loading dataset...")
dataset = load_dataset("jarvisx17/Medical-ASR-EN", split="train")

# -------------------- Step 2: Shuffle and Split --------------------
print("Shuffling and splitting dataset...")
dataset = dataset.shuffle(seed=42)
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

dataset_splits = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

# -------------------- Step 3: Preview Split Counts --------------------
print(f"Train size     : {len(dataset_splits['train'])}")
print(f"Validation size: {len(dataset_splits['validation'])}")
print(f"Test size      : {len(dataset_splits['test'])}")

# -------------------- Step 4: Push to Hugging Face Hub --------------------
# Set your Hugging Face dataset repo name here
HF_USERNAME = "Ankit9263"  # <-- CHANGE THIS
HF_DATASET_ID = "medical-asr-split"

# Login using: huggingface-cli login
print(f"Pushing to Hugging Face Hub: {HF_USERNAME}/{HF_DATASET_ID}")
dataset_splits.push_to_hub(f"{HF_USERNAME}/{HF_DATASET_ID}")

print("✅ Dataset uploaded successfully.")

if __name__ == '__main__':
    pass