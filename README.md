# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1304.978 | 18.946 | 1.917 | 96fe5ce | e26ac9f | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26179663920) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2257.811 | 27.349 | 2.565 | 96fe5ce | e26ac9f | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26179663920) |
