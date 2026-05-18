# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1302.198 | 18.662 | 2.823 | db88189 | 49f9a1d | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26031178686) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2282.178 | 33.745 | 2.825 | db88189 | 49f9a1d | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26031178686) |
