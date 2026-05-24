# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1328.602 | 19.271 | 1.901 | 21e14fb | 68a4c6b | local |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2315.715 | 28.600 | 2.603 | 21e14fb | 68a4c6b | local |

### Model Generation Report

Serial generation smoke tests validate that each model produces coherent output on Vulkan.

| Model | Output | Coherent | Peak memory (GB) | Sample | Error |
| --- | --- | --- | ---: | --- | --- |
| mlx-community/Qwen3-0.6B-bf16 | pass | pass | 1.144 | Vulkan acceleration is useful because it allows for efficient rendering of complex scenes wit... |  |
| mlx-community/Qwen3-0.6B-8bit | pass | pass | 0.956 | Vulkan acceleration is useful because it allows for efficient rendering of complex scenes wit... |  |
| mlx-community/Qwen3.5-2B-bf16 | pass | pass | 3.728 | <think> </think> Vulkan acceleration is useful because it allows applications to leverage the... |  |
| mlx-community/gemma-4-e2b-bf16 | pass | pass | 10.82 | Write one concise sentence about why Vulkan acceleration is useful. Write one concise sentenc... |  |
