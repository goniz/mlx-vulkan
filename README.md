# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1330.239 | 19.969 | 1.917 | 53a6764 | 2a18b7c | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26142834214) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2337.462 | 28.049 | 2.565 | 53a6764 | 2a18b7c | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26142834214) |
