# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=2001.901, generation_tps=9.551, peak_memory=3.329, total_time=15.569
Trial 2:  prompt_tps=2014.342, generation_tps=9.864, peak_memory=3.329, total_time=15.114
Trial 3:  prompt_tps=2004.437, generation_tps=9.793, peak_memory=3.329, total_time=15.218
Trial 4:  prompt_tps=2018.585, generation_tps=9.504, peak_memory=3.329, total_time=15.602
Trial 5:  prompt_tps=2009.968, generation_tps=9.811, peak_memory=3.330, total_time=15.188
Averages: prompt_tps=2009.847, generation_tps=9.705, peak_memory=3.329

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 40030.47it/s]```
Performance vs baseline: **prompt_tps +1.6%**, **generation_tps +58.5%**, **peak_memory -0.4%**

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1119.587, generation_tps=14.132, peak_memory=2.922, total_time=12.816
Trial 2:  prompt_tps=1117.434, generation_tps=12.803, peak_memory=2.922, total_time=13.766
Trial 3:  prompt_tps=1124.145, generation_tps=14.374, peak_memory=2.923, total_time=12.653
Trial 4:  prompt_tps=1118.746, generation_tps=14.456, peak_memory=2.923, total_time=12.618
Trial 5:  prompt_tps=1119.308, generation_tps=13.549, peak_memory=2.923, total_time=13.214
Averages: prompt_tps=1119.844, generation_tps=13.863, peak_memory=2.923

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 1386.95it/s]```
Performance vs baseline: **prompt_tps +0.9%**, **generation_tps +32.0%**, **peak_memory -0.1%**

### Qwen3.6-35B-A3B (MoE)

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1119.587, generation_tps=14.132, peak_memory=2.922, total_time=12.816
Trial 2:  prompt_tps=1117.434, generation_tps=12.803, peak_memory=2.922, total_time=13.766
Trial 3:  prompt_tps=1124.145, generation_tps=14.374, peak_memory=2.923, total_time=12.653
Trial 4:  prompt_tps=1118.746, generation_tps=14.456, peak_memory=2.923, total_time=12.618
Trial 5:  prompt_tps=1119.308, generation_tps=13.549, peak_memory=2.923, total_time=13.214
Averages: prompt_tps=1119.844, generation_tps=13.863, peak_memory=2.923

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 1386.95it/s]```
