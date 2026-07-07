# mlx-vulkan
Home for the Development of MLX Vulkan backend

The motivation for this side project is to make MLX work on my Strix Halo machine since I fall in love with MLX and the MLX Community 

## Acknowledgments

The Vulkan compute shaders in `mlx/mlx/backend/vulkan/kernels/` were originally taken from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License, copyright The ggml authors) and modified for use in mlx-vulkan.

## AI-Assisted Development

This project was developed with the assistance of LLM coding agents,
primarily GPT-5.5, under heavy human supervision and steering.

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
| mlx-community/Qwen3-0.6B-8bit | 8bit | 1492.104 | 89.988 | 1.554 | f49be61 | e639eb1 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/28719951837) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 2812.254 | 66.926 | 2.101 | f49be61 | e639eb1 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/28719951837) |
| mlx-community/Qwen3.6-35B-A3B-8bit | 8bit | 126.534 | 21.898 | 39.235 | f49be61 | e639eb1 | [run](https://github.com/goniz/mlx-vulkan/actions/runs/28719951837) |

### Model Generation Report

Serial generation smoke tests validate that each model produces coherent output on Vulkan.

| Model | Output | Coherent | Peak memory (GB) | Sample | Error |
| --- | --- | --- | ---: | --- | --- |
| mlx-community/Qwen3-0.6B-bf16 | pass | pass | 1.142 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| mlx-community/Qwen3-0.6B-8bit | pass | pass | 0.625 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit | pass | pass | 1.183 | Vulkan acceleration enhances performance by enabling efficient parallel processing and reduci... |  |
| mlx-community/Qwen3.5-2B-bf16 | pass | pass | 3.54 | Thinking Process: 1. **Analyze the Request:** * Task: Write one concise sentence. * Topic: Wh... |  |
| mlx-community/gemma-4-e2b-it-bf16 | pass | pass | 8.636 | <\|channel>thoughting process:1. **Analyze Request:** The user wants "one concise sentence" ex... |  |
| mlx-community/gemma-4-e4b-it-4bit | pass | pass | 11.601 | <\|channel>thought 1. **Analyze the request:** The user wants *one concise sentence* explainin... |  |
| mlx-community/gemma-4-26b-a4b-it-4bit | pass | pass | 21.893 | <\|channel>thought * Topic: Why Vulkan acceleration is useful. * Constraint: One concise sente... |  |
| mlx-community/Qwen3.6-35B-A3B-8bit | pass | pass | 34.411 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
| mlx-community/gpt-oss-20b-MXFP4-Q8 | pass | pass | 11.323 | <\|channel\|>analysis<\|message\|>We need to write one concise sentence about why Vulkan accelera... |  |
| mlx-community/Qwen3.6-27B-8bit | pass | pass | 26.886 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
