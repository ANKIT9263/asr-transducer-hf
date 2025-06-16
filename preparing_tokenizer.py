import os
import requests
from omegaconf import OmegaConf
import yaml
import sys

# ---------- Utility to download files ----------
def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)

# ---------- Step 1: Download scripts ----------
if not os.path.exists("scripts/get_hf_text_data.py"):
    download_file(
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/get_hf_text_data.py",
        "scripts/get_hf_text_data.py"
    )

if not os.path.exists('configs/huggingface_data_tokenizer.yaml'):
    download_file(
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/conf/huggingface_data_tokenizer.yaml",
        "configs/huggingface_data_tokenizer.yaml"
    )

# ---------- Step 2: Create train split config ----------
train_split = {
    'path': 'google/fleurs',
    'name': 'te_in',
    'split': 'train',
    'streaming': False
}
print(OmegaConf.to_yaml(train_split))

split_yaml = "split_config.yaml"
with open(split_yaml, "w") as f:
    yaml.dump([train_split], f)

# ---------- Step 3: Run get_hf_text_data.py ----------
sys.argv = [
    "get_hf_text_data.py",
    "--config-path=../configs",
    "--config-name=huggingface_data_tokenizer",
    "normalize_text=True",
    'symbols_to_keep=["."]',
    "text_key=transcription",
    "output_file=telugu_train_corpus.txt",
    f"+hf_data_cfg=@{split_yaml}"
]
exec(open("scripts/get_hf_text_data.py").read())

# ---------- Step 4: Preview the output ----------
print("\nPreview of telugu_train_corpus.txt:")
with open('telugu_train_corpus.txt', 'r') as f:
    for _ in range(5):
        print(f.readline().strip())

# ---------- Step 5: Download tokenizer script ----------
if not os.path.exists("scripts/process_asr_text_tokenizer.py"):
    download_file(
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/process_asr_text_tokenizer.py",
        "scripts/process_asr_text_tokenizer.py"
    )

# ---------- Step 6: Setup Tokenizer ----------
VOCAB_SIZE = 256
TOKENIZER_TYPE = "spe"
SPE_TYPE = "bpe"

if os.path.exists("tokenizers"):
    import shutil
    shutil.rmtree("tokenizers")
os.makedirs("tokenizers", exist_ok=True)

# ---------- Step 7: Run tokenizer script ----------
sys.argv = [
    "process_asr_text_tokenizer.py",
    "--data_file=telugu_train_corpus.txt",
    "--data_root=tokenizers",
    f"--tokenizer={TOKENIZER_TYPE}",
    f"--spe_type={SPE_TYPE}",
    "--no_lower_case",
    "--log",
    f"--vocab_size={VOCAB_SIZE}"
]
exec(open("scripts/process_asr_text_tokenizer.py").read())

# ---------- Step 8: Locate Tokenizer ----------
if TOKENIZER_TYPE == 'spe':
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "bpe"
else:
    TOKENIZER = os.path.join("tokenizers", f"tokenizer_wpe_v{VOCAB_SIZE}")
    TOKENIZER_TYPE_CFG = "wpe"

print("✅ Tokenizer is saved at:", TOKENIZER)
