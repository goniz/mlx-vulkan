# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running Qwen3-0.6B on AMD Radeon 8060S (Strix Halo):

### bf16
```
Running warmup..
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1473.762, generation_tps=11.169, peak_memory=2.062
Averages: prompt_tps=1473.762, generation_tps=11.169, peak_memory=2.062
```

### 8bit
*Benchmarks will be updated automatically by CI.*
