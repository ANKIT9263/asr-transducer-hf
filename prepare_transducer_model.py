import os
import subprocess
from omegaconf import OmegaConf

# Ensure config directory exists
os.makedirs("configs", exist_ok=True)

# Download speech-to-text fine-tune config
CONFIG_PATH = 'configs/speech_to_text_hf_finetune.yaml'
BRANCH = "main"  # or set this dynamically
if not os.path.exists(CONFIG_PATH):
    subprocess.run([
        "wget", "-P", "configs/",
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/asr/conf/asr_finetune/speech_to_text_hf_finetune.yaml"
    ])

# Load the YAML config
config = OmegaConf.load(CONFIG_PATH)

# Define tokenizer details (must match earlier tokenizer script)
TOKENIZER_TYPE = "spe"
SPE_TYPE = "bpe"
VOCAB_SIZE = 256
TOKENIZER_TYPE_CFG = "bpe" if TOKENIZER_TYPE == "spe" else "wpe"
TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")

# Set tokenizer details in config
config.model.tokenizer.update_tokenizer = True
config.model.tokenizer.dir = TOKENIZER
config.model.tokenizer.type = TOKENIZER_TYPE_CFG

print("✅ Tokenizer Config:")
print(OmegaConf.to_yaml(config.model.tokenizer))

# Normalize and symbols
config.model.train_ds.normalize_text = False
config.model.normalize_text = True
config.model.symbols_to_keep = ["."]
config.model.data_path = "google/fleurs"
config.model.data_name = "te_in"
config.model.streaming = False

# HuggingFace train_split
train_split = {
    'path': 'google/fleurs',
    'name': 'te_in',
    'split': 'train',
    'streaming': False
}

# Training config
config.model.train_ds.hf_data_cfg = train_split
config.model.train_ds.text_key = 'transcription'
config.model.train_ds.batch_size = 16  # adjust based on GPU
config.model.train_ds.normalize_text = True

# Validation config (copy + update split)
val_split = train_split.copy()
val_split["split"] = "validation"
config.model.validation_ds.hf_data_cfg = val_split
config.model.validation_ds.text_key = 'transcription'
config.model.validation_ds.normalize_text = True

# Test config (copy + update split)
test_split = train_split.copy()
test_split["split"] = "test"
config.model.test_ds.hf_data_cfg = test_split
config.model.test_ds.text_key = 'transcription'
config.model.test_ds.normalize_text = True

print("\n✅ Final Model Config:")
print(OmegaConf.to_yaml(config.model))

# Save modified config to file
updated_config_path = "configs/updated_speech_to_text_hf_finetune.yaml"
OmegaConf.save(config, updated_config_path)
print(f"\n📁 Updated config saved to: {updated_config_path}")
