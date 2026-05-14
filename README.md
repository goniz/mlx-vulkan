# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1428.549, generation_tps=12.303, peak_memory=2.429, total_time=13.401
Trial 2:  prompt_tps=1450.117, generation_tps=12.314, peak_memory=2.439, total_time=13.356
Trial 3:  prompt_tps=1452.526, generation_tps=12.311, peak_memory=2.469, total_time=13.356
Trial 4:  prompt_tps=1431.772, generation_tps=12.253, peak_memory=2.469, total_time=13.440
Trial 5:  prompt_tps=1442.559, generation_tps=12.156, peak_memory=2.469, total_time=13.503
Averages: prompt_tps=1441.105, generation_tps=12.267, peak_memory=2.455
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1052.812, generation_tps=10.776, peak_memory=1.403, total_time=15.897
Trial 2:  prompt_tps=1052.763, generation_tps=10.854, peak_memory=1.403, total_time=15.819
Trial 3:  prompt_tps=1063.258, generation_tps=10.769, peak_memory=1.404, total_time=15.867
Trial 4:  prompt_tps=1066.022, generation_tps=10.791, peak_memory=1.404, total_time=15.835
Trial 5:  prompt_tps=1063.556, generation_tps=10.812, peak_memory=1.404, total_time=15.821
Averages: prompt_tps=1059.682, generation_tps=10.801, peak_memory=1.404
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