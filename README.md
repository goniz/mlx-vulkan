# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

Results from running on AMD Radeon 8060S (Strix Halo):

### Qwen3-0.6B

#### bf16
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1710.670, generation_tps=19.845, peak_memory=2.435, total_time=8.971
Trial 2:  prompt_tps=1708.441, generation_tps=19.762, peak_memory=2.452, total_time=9.007
Trial 3:  prompt_tps=1690.612, generation_tps=19.898, peak_memory=2.456, total_time=8.984
Trial 4:  prompt_tps=1713.288, generation_tps=19.725, peak_memory=2.456, total_time=9.013
Trial 5:  prompt_tps=1725.742, generation_tps=19.731, peak_memory=2.456, total_time=8.992
Averages: prompt_tps=1709.750, generation_tps=19.792, peak_memory=2.451
```
Performance vs baseline: **prompt_tps +1.6%**, **generation_tps +58.5%**, **peak_memory -0.4%**

#### 8bit
```
Timing with prompt_tokens=4096, generation_tokens=128, batch_size=1.
Trial 1:  prompt_tps=1086.185, generation_tps=14.361, peak_memory=1.402, total_time=12.833
Trial 2:  prompt_tps=1083.812, generation_tps=14.569, peak_memory=1.402, total_time=12.694
Trial 3:  prompt_tps=1086.658, generation_tps=14.615, peak_memory=1.402, total_time=12.656
Trial 4:  prompt_tps=1085.929, generation_tps=14.157, peak_memory=1.402, total_time=12.947
Trial 5:  prompt_tps=1085.136, generation_tps=14.260, peak_memory=1.402, total_time=12.881
Averages: prompt_tps=1085.544, generation_tps=14.392, peak_memory=1.402
```
Performance vs baseline: **prompt_tps +0.9%**, **generation_tps +32.0%**, **peak_memory -0.1%**

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
