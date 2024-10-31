import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from MPSENet import MPSENet

# Set PyTorch CUDA configuration for optimal memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = MPSENet.from_pretrained("JacobLinCool/MP-SENet-DNS").to("cuda")

sample_rate = 16000  # Hz
durations = range(1, 14)  # From 1 to 13 seconds (OOM for 14+ seconds)
memory_usage = []
runtimes = []


def measure_memory_and_runtime(duration, sample_rate, model):
    """Measure memory usage and runtime for a given audio duration."""
    # Generate random audio data
    n_samples = int(duration * sample_rate)
    audio_data = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    audio_tensor = torch.tensor(audio_data, device="cuda")

    # Reset max memory stats
    torch.cuda.reset_max_memory_allocated()

    # Measure runtime of the forward pass
    start_time = time.time()
    model(audio_tensor, segment_size=99999999)
    end_time = time.time()

    # Calculate runtime and memory usage
    runtime = end_time - start_time
    max_memory = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

    # Clear GPU cache to avoid memory accumulation
    del audio_tensor
    torch.cuda.empty_cache()

    return max_memory, runtime


# Warm-up with 0.5 seconds of audio
print("Warming up the model...")
warmup_duration = 0.5  # 0.5 second
warmup_memory, warmup_runtime = measure_memory_and_runtime(
    warmup_duration, 16000, model
)
print(
    f"Warm-up completed - Memory: {warmup_memory:.2f} MB - Runtime: {warmup_runtime:.4f} sec"
)

# Run measurements for each duration
for duration in durations:
    max_memory, runtime = measure_memory_and_runtime(duration, sample_rate, model)
    print(
        f"Duration: {duration} sec - Memory: {max_memory:.2f} MB - Runtime: {runtime:.4f} sec"
    )
    memory_usage.append(max_memory)
    runtimes.append(runtime)

# Plot memory usage and runtime
plt.figure(figsize=(12, 6))

# Plot 1: Memory Usage
plt.subplot(1, 2, 1)
plt.plot(durations, memory_usage, marker="o", color="b")
plt.xlabel("Audio Duration (seconds)")
plt.ylabel("Max Memory Allocated (MB)")
plt.title("Memory Usage vs. Audio Duration")
plt.grid(True)

# Plot 2: Runtime
plt.subplot(1, 2, 2)
plt.plot(durations, runtimes, marker="o", color="r")
plt.xlabel("Audio Duration (seconds)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs. Audio Duration")
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()
