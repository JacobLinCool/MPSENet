# MPSENet

Python package of [MP-SENet](https://github.com/yxlu-0102/MP-SENet) from [Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement](https://arxiv.org/abs/2308.08926).

> This package is inference only. To train the model, please refer to the original repository.

## Installation

```bash
pip install MPSENet
```

## Usage

```python
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
```

> The best checkpoints trained by the original author are uploaded to Hugging Face's model hub: [g_best_dns](https://huggingface.co/JacobLinCool/MP-SENet-DNS) and [g_best_vb](https://huggingface.co/JacobLinCool/MP-SENet-VB)
