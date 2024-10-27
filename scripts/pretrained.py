import sys
import librosa
import soundfile as sf
from MPSENet import MPSENet

model = sys.argv[1]
filepath = sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

model = MPSENet.from_pretrained(model).to(device)
print(f"{model=}")

x, sr = librosa.load(filepath, sr=model.h.sampling_rate)
print(f"{x.shape=}, {sr=}")

y, sr, notation = model(x)
print(f"{y.shape=}, {sr=}, {notation=}")

sf.write("output.wav", y, sr)
