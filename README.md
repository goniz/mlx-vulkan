# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1665.949, generation_tps=12.419, peak_memory=2.450, total_time=12.899
Trial 2:  prompt_tps=1673.971, generation_tps=12.466, peak_memory=2.450, total_time=12.850
Trial 3:  prompt_tps=1664.858, generation_tps=12.403, peak_memory=2.450, total_time=12.914
Trial 4:  prompt_tps=1678.745, generation_tps=12.333, peak_memory=2.450, total_time=12.954
Trial 5:  prompt_tps=1671.959, generation_tps=12.392, peak_memory=2.450, total_time=12.911
Averages: prompt_tps=1671.096, generation_tps=12.403, peak_memory=2.450
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1071.398, generation_tps=10.824, peak_memory=1.403, total_time=15.781
Trial 2:  prompt_tps=1065.573, generation_tps=10.815, peak_memory=1.403, total_time=15.810
Trial 3:  prompt_tps=1069.161, generation_tps=10.878, peak_memory=1.404, total_time=15.730
Trial 4:  prompt_tps=1071.178, generation_tps=10.857, peak_memory=1.404, total_time=15.747
Trial 5:  prompt_tps=1069.500, generation_tps=10.673, peak_memory=1.404, total_time=15.950
Averages: prompt_tps=1069.362, generation_tps=10.809, peak_memory=1.404
```

### Qwen3.6-35B-A3B (MoE)

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=30.248, generation_tps=3.913, peak_memory=42.452
Trial 2:  prompt_tps=30.232, generation_tps=3.934, peak_memory=42.453
Trial 3:  prompt_tps=30.226, generation_tps=3.942, peak_memory=42.453
Trial 4:  prompt_tps=30.468, generation_tps=3.950, peak_memory=42.458
Trial 5:  prompt_tps=30.413, generation_tps=3.897, peak_memory=42.458
Averages: prompt_tps=30.317, generation_tps=3.927, peak_memory=42.455
```
