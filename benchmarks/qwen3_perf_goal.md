# Qwen3 0.6B BF16 Performance Goal

Source: user-supplied benchmark update
Reference runtime: `llama.cpp`
Reference model: `unsloth/Qwen3-0.6B-GGUF:BF16`
Reference GPU: `Radeon 8060S (Vulkan)`
Modalities: `text`

Target throughput for MLX Vulkan (`qwen3-0.6b-bf16`):
- Prefill goal (`pp512`): `523.32 t/s`
- Decode goal (`tg128`): `123.41 t/s`

Notes:
- Reported benchmark note: Vulkan reports `bf16: 0`, so the reference run is effectively using FP16 mode.
- Treat these as the current working performance targets for Qwen3 0.6B on this hardware.
