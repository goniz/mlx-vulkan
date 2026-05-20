# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1313.222 | 18.601 | 1.917 | 24c9e5a | 37d3e21 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26156115134) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2270.406 | 28.463 | 2.565 | 24c9e5a | 37d3e21 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26156115134) |
