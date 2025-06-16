import datasets
import soundfile as sf
import tempfile
import os

# Load a sample from the validation split
sample = datasets.load_dataset('google/fleurs', 'te_in', split='validation')[0]

# Save the audio array to a temporary .wav file
audio_array = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']  # should be 16000

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
    sf.write(tmpfile.name, audio_array, sample_rate)
    audio_path = tmpfile.name

print("✅ Audio sample saved at:", audio_path)
print("📝 Transcription:", sample['transcription'])

# Optional: play audio on supported platforms
try:
    import simpleaudio as sa
    wave_obj = sa.WaveObject.from_wave_file(audio_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()
except ImportError:
    print("🔇 Install `simpleaudio` to enable playback (pip install simpleaudio)")
except Exception as e:
    print("⚠️ Error playing audio:", str(e))
