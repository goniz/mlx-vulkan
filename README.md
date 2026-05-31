# mlx-vulkan
Home for the Development of MLX Vulkan backend

## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

## Qwen3-0.6B Prompt Throughput

![Qwen3-0.6B prompt TPS](benchmarks/prompt_tps.svg)

## Qwen3-0.6B Generation Throughput

![Qwen3-0.6B generation TPS](benchmarks/generation_tps.svg)

## Qwen3.6-35B-A3B Prompt Throughput

![Qwen3.6-35B-A3B prompt TPS](benchmarks/prompt_tps_qwen3_6_35b_a3b.svg)

## Qwen3.6-35B-A3B Generation Throughput

![Qwen3.6-35B-A3B generation TPS](benchmarks/generation_tps_qwen3_6_35b_a3b.svg)

### Latest Results

| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1319.221 | 25.538 | 2.204 | 8fbf5f9 | 0a10dbf | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26719038679) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2392.984 | 60.641 | 2.824 | 8fbf5f9 | 0a10dbf | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26719038679) |
| mlx-community/Qwen3.6-35B-A3B-8bit | 8bit | 54.350 | 8.022 | 42.438 | 8fbf5f9 | 0a10dbf | [run](https://github.com/goniz/mlx-vulkan/actions/runs/26719038679) |

### Model Generation Report

Serial generation smoke tests validate that each model produces coherent output on Vulkan.

| Model | Output | Coherent | Peak memory (GB) | Sample | Error |
| --- | --- | --- | ---: | --- | --- |
| mlx-community/Qwen3-0.6B-bf16 | pass | pass | 1.151 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| mlx-community/Qwen3-0.6B-8bit | pass | pass | 1.032 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit | pass | pass | 1.404 | Vulkan acceleration enhances performance by enabling efficient parallel processing and reduci... |  |
| mlx-community/Qwen3.5-2B-bf16 | pass | pass | 4.477 | Thinking Process: 1. **Analyze the Request:** * Task: Write one concise sentence. * Topic: Wh... |  |
| mlx-community/gemma-4-e2b-it-bf16 | pass | pass | 10.938 | Vulkan acceleration provides low-level, high-performance access to the GPU, enabling develope... |  |
| mlx-community/gemma-4-e4b-it-4bit | pass | pass | 7.459 | Vulkan acceleration provides a low-overhead, high-performance graphics API allowing developer... |  |
| mlx-community/gemma-4-26b-a4b-it-4bit | pass | pass | 15.372 | <\|channel>thought * Topic: Why Vulkan acceleration is useful. * Constraint: One concise sente... |  |
| mlx-community/Qwen3.6-35B-A3B-8bit | pass | pass | 36.206 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
| mlx-community/gpt-oss-20b-MXFP4-Q8 | pass | pass | 14.008 | <\|channel\|>analysis<\|message\|>We need to write one concise sentence about why Vulkan accelera... |  |
| mlx-community/Qwen3.6-27B-8bit | pass | pass | 29.066 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
