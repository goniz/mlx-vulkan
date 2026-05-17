# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=2299.034, generation_tps=32.640, peak_memory=2.824, total_time=5.848
Trial 2:  prompt_tps=2263.592, generation_tps=32.644, peak_memory=2.825, total_time=5.889
Trial 3:  prompt_tps=2281.701, generation_tps=33.176, peak_memory=2.825, total_time=5.781
Trial 4:  prompt_tps=2313.067, generation_tps=33.856, peak_memory=2.825, total_time=5.661
Trial 5:  prompt_tps=2322.023, generation_tps=33.567, peak_memory=2.826, total_time=5.689
Averages: prompt_tps=2295.883, generation_tps=33.177, peak_memory=2.825
```

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1331.364, generation_tps=20.689, peak_memory=2.823, total_time=9.368
Trial 2:  prompt_tps=1320.667, generation_tps=20.141, peak_memory=2.823, total_time=9.567
Trial 3:  prompt_tps=1327.813, generation_tps=20.845, peak_memory=2.823, total_time=9.339
Trial 4:  prompt_tps=1333.700, generation_tps=20.817, peak_memory=2.824, total_time=9.346
Trial 5:  prompt_tps=1321.006, generation_tps=20.260, peak_memory=2.824, total_time=9.522
Averages: prompt_tps=1326.910, generation_tps=20.551, peak_memory=2.823
```
