# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=2320.100, generation_tps=33.931, peak_memory=2.824, total_time=5.650
Trial 2:  prompt_tps=2268.379, generation_tps=33.769, peak_memory=2.825, total_time=5.710
Trial 3:  prompt_tps=2281.730, generation_tps=33.687, peak_memory=2.825, total_time=5.715
Trial 4:  prompt_tps=2264.766, generation_tps=33.519, peak_memory=2.825, total_time=5.740
Trial 5:  prompt_tps=2297.965, generation_tps=34.254, peak_memory=2.826, total_time=5.630
Averages: prompt_tps=2286.588, generation_tps=33.832, peak_memory=2.825```

#### 8bit
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1329.176, generation_tps=20.487, peak_memory=2.823, total_time=9.463
Trial 2:  prompt_tps=1320.142, generation_tps=20.246, peak_memory=2.823, total_time=9.534
Trial 3:  prompt_tps=1287.222, generation_tps=20.433, peak_memory=2.823, total_time=9.602
Trial 4:  prompt_tps=1329.389, generation_tps=20.654, peak_memory=2.824, total_time=9.398
Trial 5:  prompt_tps=1326.298, generation_tps=20.473, peak_memory=2.824, total_time=9.450
Averages: prompt_tps=1318.445, generation_tps=20.459, peak_memory=2.823```
