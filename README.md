# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1672.809, generation_tps=12.404, peak_memory=2.460, total_time=12.903
Trial 2:  prompt_tps=1678.351, generation_tps=12.596, peak_memory=2.460, total_time=12.734
Trial 3:  prompt_tps=1676.130, generation_tps=12.552, peak_memory=2.460, total_time=12.771
Trial 4:  prompt_tps=1704.015, generation_tps=12.464, peak_memory=2.460, total_time=12.809
Trial 5:  prompt_tps=1682.161, generation_tps=12.433, peak_memory=2.460, total_time=12.854
Averages: prompt_tps=1682.693, generation_tps=12.490, peak_memory=2.460
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1079.022, generation_tps=10.866, peak_memory=1.403, total_time=15.703
Trial 2:  prompt_tps=1069.481, generation_tps=10.901, peak_memory=1.403, total_time=15.705
Trial 3:  prompt_tps=1078.054, generation_tps=10.888, peak_memory=1.404, total_time=15.682
Trial 4:  prompt_tps=1072.831, generation_tps=10.904, peak_memory=1.404, total_time=15.679
Trial 5:  prompt_tps=1079.942, generation_tps=10.951, peak_memory=1.404, total_time=15.607
Averages: prompt_tps=1075.866, generation_tps=10.902, peak_memory=1.404
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
