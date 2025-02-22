import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track

from seasalt.salt_net import SaltNetOneStageHandler


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1
        )
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=False
        )
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
    ):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(
            1, channels[0], kernel_size=3, padding=1, bias=False
        )

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(num_ch, num_ah, expansion_factor)
                        for _ in range(num_tb)
                    ]
                )
                for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
            ]
        )
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList(
            [UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]]
        )
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(2, len(channels)))
            ]
        )
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(channels[2], num_heads[2], expansion_factor)
                        for _ in range(num_blocks[2])
                    ]
                )
            ]
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    TransformerBlock(channels[1], num_heads[1], expansion_factor)
                    for _ in range(num_blocks[1])
                ]
            )
        )
        # the channel of last one is not change
        self.decoders.append(
            nn.Sequential(
                *[
                    TransformerBlock(channels[1], num_heads[0], expansion_factor)
                    for _ in range(num_blocks[0])
                ]
            )
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(channels[1], num_heads[0], expansion_factor)
                for _ in range(num_refinement)
            ]
        )
        self.output = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](
            self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1))
        )
        out_dec2 = self.decoders[1](
            self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1))
        )
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out


restormer = Restormer()
restormer.to("mps")
restormer.load_state_dict(torch.load("./models/pytorch_restormer_5.h5", "mps"))


ours = SaltNetOneStageHandler(
    denoiser_path="./models/pytorch_transformers_instead_of_cnn_64_36.h5"
).denoiser


# %%
ours.eval()
restormer.eval()

# %%
model_parameters = filter(lambda p: p.requires_grad, restormer.parameters())
sum([np.prod(p.size()) for p in model_parameters])


# %%
model_parameters = filter(lambda p: p.requires_grad, ours.parameters())
sum([np.prod(p.size()) for p in model_parameters])

# Define image sizes
image_sizes = [64, 128, 256, 512, 1024, 2048, 4096]


# Define a function to benchmark a model
def benchmark_model(model, image_size, runs=1000):
    times = []
    memory_usages = []
    try:
        for _ in range(runs):
            if _ == 0:
                model(input_image)
            # Generate a random image of the given size
            input_image = torch.randn(1, 1, image_size, image_size)

            # Move the image to GPU if available
            if torch.cuda.is_available():
                with torch.no_grad():

                    input_image = input_image.cuda()
                    model.cuda()
                    torch.cuda.reset_peak_memory_stats()  # Reset memory statistics

                    # Measure the time taken for a forward pass
                    start_time = time.time()
                    model(input_image)
                    end_time = time.time()

                times.append(end_time - start_time)

                # Measure memory usage
                if torch.cuda.is_available():
                    memory_usages.append(torch.cuda.max_memory_allocated())
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


# Function to count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Get the number of parameters for each model
restormer_params = count_parameters(restormer)
ours_model_params = count_parameters(ours)

# Benchmark each model for each image size
results = []
for size in track(image_sizes):
    restormer_time, restormer_memory, restormer_throughput = benchmark_model(
        restormer, size
    )
    our_model_time, our_model_memory, our_model_throughput = benchmark_model(ours, size)
    results.append(
        {
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
    )

# Create a DataFrame to store the results
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file without index
df.to_excel("runtime_comparison.xlsx", index=False)

print("Benchmarking completed and results saved to runtime_comparison.xlsx")


# # Define image sizes
# image_sizes = [64, 128, 256, 512, 1024, 2024]


# # Define a function to benchmark a model on CPU
# def benchmark_model(model, image_size, runs=10):
#     times = []
#     # memory_usages = []
#     for _ in track(range(runs)):
#         # Generate a random image of the given size
#         input_image = torch.randn(1, 1, image_size, image_size)

#         # Move the image to CPU (this is redundant here but kept for structure)
#         input_image = input_image.to("mps")
#         model.to("mps")

#         # Measure the time taken for a forward pass
#         start_time = time.time()
#         with torch.no_grad():
#             model(input_image)
#         end_time = time.time()

#         times.append(end_time - start_time)

#         # Measure memory usage using `psutil` for CPU
#         # import psutil

#         # process = psutil.Process()
#         # memory_usages.append(process.memory_info().rss)

#     avg_time = np.mean(times)
#     # avg_memory = np.mean(memory_usages) if memory_usages else None
#     throughput = 1 / avg_time if avg_time > 0 else None

#     return avg_time, throughput


# # Function to count the number of parameters in a model
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())


# # Get the number of parameters for each model
# restormer_params = count_parameters(restormer)
# ours_model_params = count_parameters(ours)

# # Benchmark each model for each image size
# results = []
# for size in image_sizes:
#     print(size)
#     restormer_time, restormer_throughput = benchmark_model(restormer, size)
#     our_model_time, our_model_throughput = benchmark_model(ours, size)
#     results.append(
#         {
#             "Image Size": size,
#             "Restormer Time (s)": restormer_time,
#             "Our Model Time (s)": our_model_time,
#             "Restormer Throughput (images/s)": restormer_throughput,
#             "Our Model Throughput (images/s)": our_model_throughput,
#             "Restormer Params": restormer_params,
#             "Our Model Params": ours_model_params,
#         }
#     )

# # Create a DataFrame to store the results
# df = pd.DataFrame(results)

# # Save the DataFrame to an Excel file without index
# df.to_excel("runtime_comparison_cpu_mps.xlsx", index=False)

# print("Benchmarking completed and results saved to runtime_comparison_cpu.xlsx")

# with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
# ) as prof:
#     with torch.no_grad():
#         model(input_image)

# prof.step()
# times.append(prof.key_averages().self_cpu_time_total)
# memory_usages.append(prof.key_averages().self_cuda_memory_usage)
