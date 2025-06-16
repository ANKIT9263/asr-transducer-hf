if not os.path.exists("scripts/get_hf_text_data.py"):
    os.makedirs("scripts", exist_ok=True)
    subprocess.run(["wget", "-P", "scripts/",
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/get_hf_text_data.py"])
#
# # Create config dictionary
train_split = {
    'path': 'google/fleurs',
    'name': 'te_in',
    'split': 'train',
    'streaming': False
}
print(OmegaConf.to_yaml(train_split))

# # Download tokenizer config YAML
if not os.path.exists('configs/huggingface_data_tokenizer.yaml'):
    os.makedirs("configs", exist_ok=True)
    subprocess.run(["wget", "-P", "configs/",
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/conf/huggingface_data_tokenizer.yaml"])

# # Write train_split config to temporary YAML file
import yaml
# # Write train_split config to temporary YAML file (as a dict, not a list)
split_yaml = "split_config.yaml"
with open(split_yaml, "w") as f:
    yaml.dump(train_split, f)