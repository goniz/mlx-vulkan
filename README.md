# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1332.666 | 25.276 | 2.233 | f2b5585 | a027993 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26704213130) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2331.790 | 63.794 | 2.824 | f2b5585 | a027993 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26704213130) |
