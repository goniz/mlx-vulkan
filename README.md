# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1495.244, generation_tps=12.723, peak_memory=2.447, total_time=12.931
Trial 2:  prompt_tps=1482.606, generation_tps=12.915, peak_memory=2.448, total_time=12.807
Trial 3:  prompt_tps=1489.989, generation_tps=12.348, peak_memory=2.448, total_time=13.253
Trial 4:  prompt_tps=1462.652, generation_tps=13.076, peak_memory=2.448, total_time=12.724
Trial 5:  prompt_tps=1483.201, generation_tps=12.986, peak_memory=2.453, total_time=12.755
Averages: prompt_tps=1482.738, generation_tps=12.810, peak_memory=2.449
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1065.245, generation_tps=9.667, peak_memory=1.390, total_time=17.217
Trial 2:  prompt_tps=1047.892, generation_tps=9.905, peak_memory=1.390, total_time=16.962
Trial 3:  prompt_tps=1074.759, generation_tps=11.203, peak_memory=1.390, total_time=15.395
Trial 4:  prompt_tps=1065.049, generation_tps=11.219, peak_memory=1.391, total_time=15.383
Trial 5:  prompt_tps=1070.078, generation_tps=11.244, peak_memory=1.391, total_time=15.343
Averages: prompt_tps=1064.605, generation_tps=10.648, peak_memory=1.390
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