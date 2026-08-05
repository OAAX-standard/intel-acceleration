# Hardware / Precision Compatibility Matrix

## Toolchain config.json

| Precision | CPU | GPU | NPU | config.json |
|-----------|-----|-----|-----|-------------|
| **FP32** | ✅ Best accuracy | ✅ Works | ⚠️ Cast to FP16 internally — use FP16 directly | `{ "optimization": { "fp16_compression": false } }` |
| **FP16** | ⚠️ Upcasted to FP32 at runtime — no speed gain | ✅ **Recommended** (~1.5–2x vs FP32, native FP16 units) | ✅ Recommended when no calibration data | `{ "optimization": { "fp16_compression": true } }` |
| **INT8** | ✅ **Recommended** (~2x vs FP32, asymmetric preserves accuracy) | ✅ Good (asymmetric preserves accuracy) | ⚠️ Works but suboptimal — use INT8_NPU | `{ "optimization": { "fp16_compression": false, "quantization": { "enabled": true, "target_device": "any", "preset": "mixed" } } }` |
| **INT8_NPU** | ⚠️ Same as INT8 on CPU (no benefit) | ⚠️ Same as INT8 on GPU (no benefit) | ✅ **Recommended** (symmetric preset + IgnoredScope enables full NPU compiler fusion) | `{ "optimization": { "fp16_compression": false, "quantization": { "enabled": true, "target_device": "npu" } } }` |
| **FP32_U8** | ✅ Convenience (feed raw pixels, same accuracy as FP32) | ✅ Convenience | ✅ Convenience | `{ "optimization": { "fp16_compression": false }, "preprocessing": { "input_dtype": "u8", "scale_values": [255.0, 255.0, 255.0] } }` |

> **INT8_NPU note:** `target_device: "npu"` automatically forces `preset: "performance"` and applies an IgnoredScope keeping `attn/*`, `dfl/*`, and `MatMul` ops in FP16.

---

## Runtime config keys

| Key | CPU | GPU | NPU |
|-----|-----|-----|-----|
| `device_type` | `"CPU"` | `"GPU"` / `"GPU.0"` | `"NPU"` |
| `perf_hint` | `"throughput"` | `"throughput"` (single GPU) / `"cumulative_throughput"` (MULTI) | `"throughput"` |
| `num_streams` | `"0"` (auto) | `"4"` recommended for discrete GPU | `"0"` (NPU ignores it) |
| `auto_batch_size` | ❌ Ignored | `"4"`–`"8"` (iGPU); avoid on NVIDIA OpenCL (degrades >2) | ❌ Not applicable |
| `npu_turbo` | ❌ Warning logged | ❌ Warning logged | `"1"` if driver supports it |

### Recommended runtime configs

**CPU — INT8, latency-optimised:**
```json
{
  "device_type": "CPU",
  "perf_hint": "throughput"
}
```

**GPU — FP16, throughput-optimised:**
```json
{
  "device_type": "GPU",
  "perf_hint": "throughput",
  "num_streams": "4"
}
```

**GPU — multi-GPU:**
```json
{
  "device_type": "GPU",
  "perf_hint": "cumulative_throughput",
  "num_streams": "4"
}
```

**NPU — INT8_NPU:**
```json
{
  "device_type": "NPU",
  "perf_hint": "throughput",
  "npu_turbo": "1"
}
```

---

## Quick reference: best precision per device

| Device | Best precision | Input dtype | Rationale |
|--------|---------------|-------------|-----------|
| CPU | INT8 (`target_device: any`, `preset: mixed`) | `u8` | ~2x throughput; asymmetric quantization preserves accuracy |
| GPU | FP16 or INT8 (`target_device: any`, `preset: mixed`) | `f16` / `u8` | Native FP16 units; INT8 for max throughput |
| NPU | INT8_NPU (`target_device: npu`) | `u8` | Only variant enabling full NPU compiler fusion |
| NPU (no calib) | FP16 | `f16` | NPU runs FP16 natively; FP32 offers no benefit |
