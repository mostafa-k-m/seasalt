import time

import numpy as np
import pandas as pd
import psutil
import torch
from rich.progress import track

from restormer_implementation import Restormer
from seasalt.salt_net import SaltNetOneStageHandler

device = "mps"
max_image_size = 4096
min_image_size = 64
exp_name = f"refactor_{min_image_size}_{max_image_size}_{device}"

image_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

restormer = Restormer()
restormer.to(device)
restormer.load_state_dict(torch.load("./models/pytorch_restormer_5.h5", device))


ours = SaltNetOneStageHandler(
    denoiser_path="./models/pytorch_lightweight_model_sigmoid_transferred_125.h5",
    fallback_device=device,
).denoiser


ours.eval()
restormer.eval()

model_parameters = filter(lambda p: p.requires_grad, restormer.parameters())
model_parameters = filter(lambda p: p.requires_grad, ours.parameters())


def benchmark_model(model, image_size, runs=1000):
    times = []
    memory_usages = []
    input_image = torch.randn(1, 1, image_size, image_size).to(device)
    try:
        for _ in range(runs):
            if torch.cuda.is_available():
                with torch.no_grad():

                    input_image = input_image.cuda()
                    model.cuda()
                    torch.cuda.reset_peak_memory_stats()

                    start_time = time.time()
                    model(input_image)
                    end_time = time.time()

                times.append(end_time - start_time)
                if torch.cuda.is_available():
                    memory_usages.append(torch.cuda.max_memory_allocated())
            else:
                start_time = time.time()
                with torch.no_grad():
                    model(input_image)
                end_time = time.time()

                times.append(end_time - start_time)

                if device == "cpu":
                    process = psutil.Process()
                    memory_usages.append(process.memory_info().rss)
                elif device == "mps":
                    memory_usages.append(torch.mps.current_allocated_memory())
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usages) if memory_usages else None
        throughput = 1 / avg_time if avg_time > 0 else None
    except torch.cuda.OutOfMemoryError:
        return (
            "Not completed, Cuda ran out of memory",
            "Not completed, Cuda ran out of memory",
            "Not completed, Cuda ran out of memory",
        )
    return avg_time, avg_memory, throughput


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


restormer_params = count_parameters(restormer)
ours_model_params = count_parameters(ours)

results = []
for size in track(image_sizes):
    if size > max_image_size or size < min_image_size:
        continue
    print(f"Image Size: {size}")
    restormer_time, restormer_memory, restormer_throughput = benchmark_model(
        restormer, size
    )
    our_model_time, our_model_memory, our_model_throughput = benchmark_model(ours, size)
    result_dict = {
        "Image Size": size,
        "Restormer Time (s)": restormer_time,
        "Our Model Time (s)": our_model_time,
        "Restormer Memory (bytes)": restormer_memory,
        "Our Model Memory (bytes)": our_model_memory,
        "Restormer Throughput (images/s)": restormer_throughput,
        "Our Model Throughput (images/s)": our_model_throughput,
        "Restormer Params": restormer_params,
        "Our Model Params": ours_model_params,
    }
    print(result_dict)
    results.append(result_dict)

df = pd.DataFrame(results)

df.to_excel(f"runtime_comparison_{exp_name}.xlsx", index=False)

print(f"Benchmarking completed and results saved to runtime_comparison_{exp_name}.xlsx")
