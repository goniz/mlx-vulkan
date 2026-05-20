# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1266.470 | 17.653 | 1.917 | 33eb2d2 | 13c3096 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26154299689) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2300.015 | 27.642 | 2.565 | 33eb2d2 | 13c3096 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26154299689) |
