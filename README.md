# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1320.984 | 18.960 | 1.896 | 545e44c | 045d17b | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26368847068) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2286.532 | 28.557 | 2.603 | 545e44c | 045d17b | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26368847068) |
