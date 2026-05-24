# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1329.371 | 25.035 | 2.204 | ea6f0f4 | 8e4a5d8 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26375042081) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2335.527 | 62.190 | 2.824 | ea6f0f4 | 8e4a5d8 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26375042081) |

### Model Generation Report

Serial generation smoke tests validate that each model produces coherent output on Vulkan.

| Model | Output | Coherent | Peak memory (GB) | Sample | Error |
| --- | --- | --- | ---: | --- | --- |
| mlx-community/Qwen3-0.6B-bf16 | pass | pass | 1.146 | Vulkan acceleration is useful because it allows for efficient rendering of complex scenes wit... |  |
| mlx-community/Qwen3-0.6B-8bit | pass | pass | 1.033 | Vulkan acceleration is useful because it allows for efficient rendering of complex scenes wit... |  |
| mlx-community/Qwen3.5-2B-bf16 | pass | pass | 3.751 | <think> </think> Vulkan acceleration is useful because it allows applications to leverage the... |  |
| mlx-community/gemma-4-e2b-bf16 | pass | pass | 10.927 | Write one concise sentence about why Vulkan acceleration is useful. Write one concise sentenc... |  |
