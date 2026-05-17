# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1996.983, generation_tps=9.491, peak_memory=3.329, total_time=15.646
Trial 2:  prompt_tps=1930.992, generation_tps=9.445, peak_memory=3.329, total_time=15.868
Trial 3:  prompt_tps=2018.265, generation_tps=9.772, peak_memory=3.329, total_time=15.230
Trial 4:  prompt_tps=2003.143, generation_tps=9.675, peak_memory=3.329, total_time=15.399
Trial 5:  prompt_tps=2001.867, generation_tps=9.918, peak_memory=3.330, total_time=15.055
Averages: prompt_tps=1990.250, generation_tps=9.660, peak_memory=3.329

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 33885.76it/s]```
Performance vs baseline: **prompt_tps +1.6%**, **generation_tps +58.5%**, **peak_memory -0.4%**

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1119.405, generation_tps=14.335, peak_memory=2.922, total_time=12.696
Trial 2:  prompt_tps=1119.787, generation_tps=13.191, peak_memory=2.922, total_time=13.468
Trial 3:  prompt_tps=1123.557, generation_tps=14.061, peak_memory=2.923, total_time=12.854
Trial 4:  prompt_tps=1117.433, generation_tps=13.464, peak_memory=2.923, total_time=13.281
Trial 5:  prompt_tps=1124.360, generation_tps=12.944, peak_memory=2.923, total_time=13.640
Averages: prompt_tps=1120.908, generation_tps=13.599, peak_memory=2.923

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 240437.81it/s]```
Performance vs baseline: **prompt_tps +0.9%**, **generation_tps +32.0%**, **peak_memory -0.1%**

### Qwen3.6-35B-A3B (MoE)

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1119.405, generation_tps=14.335, peak_memory=2.922, total_time=12.696
Trial 2:  prompt_tps=1119.787, generation_tps=13.191, peak_memory=2.922, total_time=13.468
Trial 3:  prompt_tps=1123.557, generation_tps=14.061, peak_memory=2.923, total_time=12.854
Trial 4:  prompt_tps=1117.433, generation_tps=13.464, peak_memory=2.923, total_time=13.281
Trial 5:  prompt_tps=1124.360, generation_tps=12.944, peak_memory=2.923, total_time=13.640
Averages: prompt_tps=1120.908, generation_tps=13.599, peak_memory=2.923

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 240437.81it/s]```
