# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1311.879 | 18.768 | 1.901 | 21e14fb | 68a4c6b | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26371082381) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2264.269 | 28.568 | 2.603 | 21e14fb | 68a4c6b | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26371082381) |
