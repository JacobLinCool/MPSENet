import os
import sys
from datasets import DatasetDict, Dataset, Audio

base_dir = sys.argv[1]

wav_clean_dir = os.path.join(base_dir, "wav_clean")
wav_noisy_dir = os.path.join(base_dir, "wav_noisy")
training_txt = os.path.join(base_dir, "training.txt")
test_txt = os.path.join(base_dir, "test.txt")


def read_txt(txt):
    with open(txt, "r") as f:
        lines = f.readlines()

    return [line.split("|")[0] for line in lines]


training_ids = read_txt(training_txt)
test_ids = read_txt(test_txt)
print(f"Training: {len(training_ids)}")
print(f"Test: {len(test_ids)}")

training_data = []
test_data = []

for id in training_ids:
    clean = os.path.join(wav_clean_dir, f"{id}.wav")
    noisy = os.path.join(wav_noisy_dir, f"{id}.wav")
    training_data.append({"id": id, "clean": clean, "noisy": noisy})

for id in test_ids:
    clean = os.path.join(wav_clean_dir, f"{id}.wav")
    noisy = os.path.join(wav_noisy_dir, f"{id}.wav")
    test_data.append({"id": id, "clean": clean, "noisy": noisy})

ds = DatasetDict()

ds["train"] = Dataset.from_list(training_data)
ds["test"] = Dataset.from_list(test_data)

ds = ds.cast_column("clean", Audio(sampling_rate=16000))
ds = ds.cast_column("noisy", Audio(sampling_rate=16000))

ds.push_to_hub("JacobLinCool/VoiceBank-DEMAND-16k")
