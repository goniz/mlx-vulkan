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
| mlx-community/Qwen3-0.6B-8bit | 8bit | 4124.938 | 119.797 | 1.998 | e6b2da3 | ea5289c | [run](https://github.com/goniz/mlx-vulkan/actions/runs/29495909657) |
| mlx-community/Qwen3-0.6B-bf16 | bf16 | 4178.176 | 67.994 | 2.262 | e6b2da3 | ea5289c | [run](https://github.com/goniz/mlx-vulkan/actions/runs/29495909657) |
| mlx-community/Qwen3.6-35B-A3B-8bit | 8bit | 903.405 | 29.568 | 39.885 | e6b2da3 | ea5289c | [run](https://github.com/goniz/mlx-vulkan/actions/runs/29495909657) |

### Model Generation Report

Serial generation smoke tests validate that each model produces coherent output on Vulkan.

| Model | Output | Coherent | Peak memory (GB) | Sample | Error |
| --- | --- | --- | ---: | --- | --- |
| mlx-community/Qwen3-0.6B-bf16 | pass | pass | 1.143 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| mlx-community/Qwen3-0.6B-8bit | pass | pass | 0.624 | <think> Okay, the user wants a concise sentence about why Vulkan acceleration is useful. Let... |  |
| LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit | pass | pass | 1.183 | Vulkan acceleration enhances performance by enabling efficient parallel processing and reduci... |  |
| mlx-community/Qwen3.5-2B-bf16 | pass | pass | 3.541 | Thinking Process: 1. **Analyze the Request:** * Task: Write one concise sentence. * Topic: Wh... |  |
| mlx-community/gemma-4-e2b-it-bf16 | pass | pass | 8.639 | <\|channel>thoughting process:1. **Analyze Request:** The user wants "one concise sentence" ex... |  |
| mlx-community/gemma-4-e4b-it-4bit | pass | pass | 3.968 | <\|channel>thought 1. **Analyze the request:** The user wants *one concise sentence* explainin... |  |
| mlx-community/gemma-4-26b-a4b-it-4bit | pass | pass | 13.308 | <\|channel>thought * Topic: Why Vulkan acceleration is useful. * Constraint: One concise sente... |  |
| mlx-community/Qwen3.6-35B-A3B-8bit | pass | pass | 34.41 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
| mlx-community/gpt-oss-20b-MXFP4-Q8 | pass | pass | 11.436 | <\|channel\|>analysis<\|message\|>We need to write one concise sentence about why Vulkan accelera... |  |
| mlx-community/Qwen3.6-27B-8bit | pass | pass | 26.871 | Here's a thinking process: 1. **Analyze User Input:** - **Topic:** Vulkan acceleration - **Re... |  |
