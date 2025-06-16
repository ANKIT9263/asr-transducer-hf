import os
import datasets
import soundfile as sf
from nemo.collections.asr.models import ASRModel

# -------------------- Load Fine-Tuned Model --------------------
print("🔁 Restoring fine-tuned ASR model from file...")
asr_model = ASRModel.restore_from("telugu_asr_model.nemo")
asr_model.eval()

# -------------------- Delete .nemo file to save space --------------------
if os.path.exists("telugu_asr_model.nemo"):
    os.remove("telugu_asr_model.nemo")
    print("🧹 Deleted telugu_asr_model.nemo to save space.")

# -------------------- Load a Test Sample --------------------
print("📥 Loading test sample from FLEURS (te_in)...")
sample = datasets.load_dataset('google/fleurs', 'te_in', split='test')[0]

# Save audio to disk
audio_array = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']
audio_path = "sample.wav"
sf.write(audio_path, audio_array, sample_rate)

# -------------------- Run Inference --------------------
print("🔊 Transcribing sample...")
predicted_transcription = asr_model.transcribe([audio_path])[0]

# Print Result
print("\n🎤 Predicted Transcription:", predicted_transcription)
print("📘 Target Transcription   :", sample['transcription'])

# -------------------- Cleanup --------------------
os.remove(audio_path)
print("🧹 Removed temporary audio file.")
