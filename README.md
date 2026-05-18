# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=2306.928, generation_tps=33.816, peak_memory=2.824, total_time=5.670
Trial 2:  prompt_tps=2268.826, generation_tps=33.681, peak_memory=2.825, total_time=5.716
Trial 3:  prompt_tps=2287.636, generation_tps=32.873, peak_memory=2.825, total_time=5.798
Trial 4:  prompt_tps=2312.493, generation_tps=33.633, peak_memory=2.825, total_time=5.692
Trial 5:  prompt_tps=2303.616, generation_tps=33.083, peak_memory=2.826, total_time=5.816
Averages: prompt_tps=2295.900, generation_tps=33.417, peak_memory=2.825```

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1331.335, generation_tps=20.855, peak_memory=2.823, total_time=9.316
Trial 2:  prompt_tps=1297.875, generation_tps=20.677, peak_memory=2.823, total_time=9.454
Trial 3:  prompt_tps=1327.138, generation_tps=20.095, peak_memory=2.823, total_time=9.585
Trial 4:  prompt_tps=1315.170, generation_tps=20.473, peak_memory=2.824, total_time=9.475
Trial 5:  prompt_tps=1315.248, generation_tps=19.891, peak_memory=2.824, total_time=9.662
Averages: prompt_tps=1317.353, generation_tps=20.398, peak_memory=2.823```
