# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1456.303, generation_tps=11.099, peak_memory=1.860
Trial 2:  prompt_tps=1465.758, generation_tps=11.080, peak_memory=1.860
Trial 3:  prompt_tps=1462.110, generation_tps=11.124, peak_memory=1.861
Trial 4:  prompt_tps=1462.941, generation_tps=11.204, peak_memory=1.861
Trial 5:  prompt_tps=1462.306, generation_tps=11.153, peak_memory=1.861
Averages: prompt_tps=1461.884, generation_tps=11.132, peak_memory=1.861
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1082.887, generation_tps=8.853, peak_memory=1.302
Trial 2:  prompt_tps=1081.689, generation_tps=8.913, peak_memory=1.303
Trial 3:  prompt_tps=1088.131, generation_tps=8.818, peak_memory=1.303
Trial 4:  prompt_tps=1088.904, generation_tps=8.849, peak_memory=1.303
Trial 5:  prompt_tps=1087.970, generation_tps=8.686, peak_memory=1.304
Averages: prompt_tps=1085.916, generation_tps=8.824, peak_memory=1.303
```

### Qwen3.6-35B-A3B (MoE)

TBD