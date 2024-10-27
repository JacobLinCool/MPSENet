import sys
import torch
from MPSENet import MPSENet, AttrDict

filepath = sys.argv[1]
repo = sys.argv[2]

h = AttrDict(
    {
        "dense_channel": 64,
        "compress_factor": 0.3,
        "num_tsconformers": 4,
        "beta": 2.0,
        "sampling_rate": 16000,
        "segment_size": 32000,
        "n_fft": 400,
        "hop_size": 100,
        "win_size": 400,
    }
)

model = MPSENet(h)
state = torch.load(filepath, map_location="cpu", weights_only=True)
model.load_state_dict(state["generator"])

model.push_to_hub(repo)
