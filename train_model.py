import os
# os.environ["NUMBA_CUDA_DEFAULT_PTX_CC"] = "8.4"
# os.environ["CUDA_ENABLE_PYNVJITLINK"] = "0"
# os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from lightning.pytorch import Trainer
import nemo.collections.asr as nemo_asr
# -------------------- Load Config --------------------
config_path = "configs/updated_speech_to_text_hf_finetune.yaml"
config = OmegaConf.load(config_path)

# -------------------- Optimizer & Scheduler --------------------
config.model.optim.name = "adamw"
config.model.optim.lr = 3e-4
config.model.optim.betas = [0.9, 0.98]
config.model.optim.weight_decay = 0.001

config.model.optim.sched.name = "CosineAnnealing"
config.model.optim.sched.warmup_steps = 500  # 10% of 5000 steps
config.model.optim.sched.min_lr = 5e-6
config.model.optim.sched.warmup_ratio = None

# -------------------- Remove SpecAugment --------------------
if "spec_augment" in config.model:
    del config.model.spec_augment
    print("❌ Removed spec_augment from config")

# -------------------- Load Pretrained Model --------------------
print("📦 Loading pre-trained model: nvidia/parakeet-tdt-0.6b-v2")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# -------------------- Update Tokenizer --------------------
TOKENIZER_TYPE = "spe"
SPE_TYPE = "bpe"
VOCAB_SIZE = 256
TOKENIZER_TYPE_CFG = "bpe"
TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")

decoder_state = asr_model.decoder.state_dict()
joint_state = asr_model.joint.state_dict()
prev_vocab_size = asr_model.tokenizer.vocab_size

asr_model.change_vocabulary(new_tokenizer_dir=TOKENIZER, new_tokenizer_type=TOKENIZER_TYPE_CFG)

if asr_model.tokenizer.vocab_size == prev_vocab_size:
    asr_model.decoder.load_state_dict(decoder_state)
    asr_model.joint.load_state_dict(joint_state)
    print("✅ Loaded previous decoder and joint state (vocab size unchanged)")

# -------------------- Setup Dataloaders --------------------
cfg = model_utils.convert_model_config_to_dict_config(config)

print("📂 Setting up training data...")
asr_model.setup_training_data(cfg.model.train_ds)

print("📂 Setting up validation data...")
asr_model.setup_validation_data(cfg.model.validation_ds)

if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.get("hf_data_cfg", {}).get("split") == "test":
    print("📂 Setting up test data...")
    asr_model.setup_test_data(cfg.model.test_ds)

# -------------------- Setup Optimizer --------------------
print("⚙️ Setting up optimizer...")
asr_model.setup_optimization(cfg.model.optim)

# -------------------- Trainer Init --------------------
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 5000

trainer = Trainer(
    devices=1,
    accelerator=accelerator,
    max_epochs=-1,
    max_steps=MAX_STEPS,
    enable_checkpointing=False,
    logger=False,
    log_every_n_steps=100,
    check_val_every_n_epoch=10,
    precision='bf16' if torch.cuda.is_bf16_supported() else 16,
)

# -------------------- Start Training --------------------
print("🚀 Starting training...")
trainer.fit(asr_model)

# -------------------- Save Fine-tuned Model --------------------
output_model_path = "telugu_asr_model.nemo"
asr_model.save_to(output_model_path)
print(f"✅ Model saved to: {output_model_path}")
