from gtts import gTTS
import os

texts = [
    "Hello, this is a test audio.",
    "I am speaking in a natural way.",
    "This is generated speech.",
    "Fake audio detection is important.",
    "Machine learning can help identify deepfakes.",
    "Voice cloning is advancing rapidly.",
    "Artificial intelligence is changing the world.",
    "Please verify the authenticity of this audio.",
    "This is a sample fake voice.",
    "Detection of synthetic speech is crucial."
]

os.makedirs('data/fake', exist_ok=True)

for i, text in enumerate(texts):
    tts = gTTS(text=text, lang='en')
    tts.save(f'data/fake/fake_{i}.mp3')