# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1154.232, generation_tps=9.570, peak_memory=1.860
Trial 2:  prompt_tps=1155.190, generation_tps=10.312, peak_memory=1.860
Trial 3:  prompt_tps=1656.567, generation_tps=18.183, peak_memory=1.861
Trial 4:  prompt_tps=1676.066, generation_tps=18.148, peak_memory=1.861
Trial 5:  prompt_tps=1688.025, generation_tps=18.186, peak_memory=1.861
Averages: prompt_tps=1466.016, generation_tps=14.880, peak_memory=1.861
```

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1048.982, generation_tps=7.524, peak_memory=1.317
Trial 2:  prompt_tps=1038.437, generation_tps=7.465, peak_memory=1.317
Trial 3:  prompt_tps=1047.659, generation_tps=7.493, peak_memory=1.317
Trial 4:  prompt_tps=1043.262, generation_tps=7.487, peak_memory=1.317
Trial 5:  prompt_tps=1045.936, generation_tps=7.419, peak_memory=1.327
Averages: prompt_tps=1044.855, generation_tps=7.478, peak_memory=1.319
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