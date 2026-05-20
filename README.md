# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1317.869 | 18.364 | 1.919 | 59b4efe | 7e13fc8 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26177458763) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2289.089 | 28.488 | 2.565 | 59b4efe | 7e13fc8 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26177458763) |
