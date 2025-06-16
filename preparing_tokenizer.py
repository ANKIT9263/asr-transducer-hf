import os
import subprocess
from omegaconf import OmegaConf

# Download get_hf_text_data.py if not exists
if not os.path.exists("scripts/get_hf_text_data.py"):
    os.makedirs("scripts", exist_ok=True)
    subprocess.run(["wget", "-P", "scripts/",
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/get_hf_text_data.py"])

# Create config dictionary
train_split = {
    'path': 'google/fleurs',
    'name': 'te_in',
    'split': 'train',
    'streaming': False
}
print(OmegaConf.to_yaml(train_split))

# Download tokenizer config YAML
if not os.path.exists('configs/huggingface_data_tokenizer.yaml'):
    os.makedirs("configs", exist_ok=True)
    subprocess.run(["wget", "-P", "configs/",
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/conf/huggingface_data_tokenizer.yaml"])

# Write train_split config to temporary YAML file
import yaml
split_yaml = "split_config.yaml"
with open(split_yaml, "w") as f:
    yaml.dump([train_split], f)

# Run data extraction script
subprocess.run([
    "python", "scripts/get_hf_text_data.py",
    "--config-path=../configs",
    "--config-name=huggingface_data_tokenizer",
    "normalize_text=True",
    'symbols_to_keep=["."]',
    "text_key=transcription",
    "output_file=telugu_train_corpus.txt",
    f"+hf_data_cfg=@{split_yaml}"
])

# Preview first 5 lines
print("\nPreview of telugu_train_corpus.txt:")
with open('telugu_train_corpus.txt', 'r') as f:
    for _ in range(5):
        print(f.readline().strip())

# Download tokenizer script if not exists
if not os.path.exists("scripts/process_asr_text_tokenizer.py"):
    branch = "main"
    subprocess.run([
        "wget", "-P", "scripts/",
        f"https://raw.githubusercontent.com/NVIDIA/NeMo/{branch}/scripts/tokenizers/process_asr_text_tokenizer.py"
    ])

# Setup tokenizer variables
VOCAB_SIZE = 256
TOKENIZER_TYPE = "spe"
SPE_TYPE = "bpe"

# Clean and prepare tokenizer directory
if os.path.exists("tokenizers"):
    subprocess.run(["rm", "-r", "tokenizers"])
os.makedirs("tokenizers", exist_ok=True)

# Run tokenizer training script
subprocess.run([
    "python", "scripts/process_asr_text_tokenizer.py",
    "--data_file=telugu_train_corpus.txt",
    "--data_root=tokenizers",
    f"--tokenizer={TOKENIZER_TYPE}",
    f"--spe_type={SPE_TYPE}",
    "--no_lower_case",
    "--log",
    f"--vocab_size={VOCAB_SIZE}"
])

# Determine tokenizer path
if TOKENIZER_TYPE == 'spe':
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "bpe"
else:
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_wpe_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "wpe"

print("✅ Tokenizer is saved at:", TOKENIZER)
