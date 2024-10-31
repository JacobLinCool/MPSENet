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

x, sr = librosa.load(filepath, sr=model.sampling_rate)
print(f"{x.shape=}, {sr=}")

y, sr, notation = model(x)
print(f"{y.shape=}, {sr=}, {notation=}")

sf.write("output.wav", y, sr)
```

> The best checkpoints trained by the original author are uploaded to Hugging Face's model hub: [g_best_dns](https://huggingface.co/JacobLinCool/MP-SENet-DNS) and [g_best_vb](https://huggingface.co/JacobLinCool/MP-SENet-VB)

## Memory Usage and Speed

By default, the model will chunk the input audio into 2-second segments and process them one by one. This is to prevent memory overflow and allow the model to run on almost any machine out of the box.

If you have enough memory, you can set `segment_size` to a larger value (e.g., 160,000 for 10 seconds), which may help to generate better results in some cases.

![Memory Usage and Speed](images/usage.jpg)

| Segment Length (sec) | Memory (MB) | Runtime (sec) |
| -------------------- | ----------- | ------------- |
| 1                    | 398.38      | 0.1234        |
| 2                    | 769.42      | 0.2051        |
| 3                    | 1161.13     | 0.2439        |
| 4                    | 1573.53     | 0.3538        |
| 5                    | 2249.46     | 0.4741        |
| 6                    | 3190.23     | 0.6266        |
| 7                    | 4296.47     | 0.8872        |
| 8                    | 5568.20     | 1.0642        |
| 9                    | 7005.40     | 1.2646        |
| 10                   | 8608.08     | 1.4580        |
| 11                   | 10376.24    | 1.6912        |
| 12                   | 12309.88    | 1.9670        |
| 13                   | 14408.99    | 2.2056        |

> Tested on Google Colab with a Tesla T4 GPU. See [scripts/benchmark.py](scripts/benchmark.py) for more details.
