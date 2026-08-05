# NPU / iGPU Performance Notes (WCL)

Remarks from hardware benchmarking on Intel WCL (Lunar Lake / Arrow Lake class).

---

## 1. NPU does not always outperform iGPU — expected

Performance winner is workload-dependent; both directions are normal:

| Scenario | Winner | Reason |
|----------|--------|--------|
| Conv-heavy graphs, moderate-to-large inputs, sustained tensor work | NPU | NPU MAC pipeline is well-utilized |
| Very small inputs / very small models | iGPU | NPU compile and DMA setup overhead dominates actual compute |
| Graphs with significant attention (Softmax / MatMul) | iGPU | NPU compiler cannot fuse those ops into preferred INT8 kernel families |

**Observed example:** `yolo11n_320` reaches ~1357 FPS on NPU at FP16 (NPU wins decisively), but `yolo11s` INT8 favors iGPU because attention overhead is a larger fraction of total compute.

---

## 2. FP16 ≈ FP32 on NPU — expected

This NPU generation has no native FP32 MAC pipeline. The compiler internally downcasts FP32 graphs to FP16, so FP32 and FP16 variants benchmark the same compiled graph. Observed difference is within ~1% across all model/device combinations — the expected signature of this behavior.

---

## 3. INT8 slower than FP32 on NPU — root cause + fix

### Root cause

The toolchain was calling `nncf.quantize()` without:
- `target_device=nncf.TargetDevice.NPU`
- An `IgnoredScope` excluding attention / DFL subgraphs

Without these, the NPU compiler inserts `FakeQuantize → Convert → FP16-compute → FakeQuantize` boundaries around every Softmax / MatMul it cannot fuse into INT8. On YOLO architectures with attention blocks (YOLO11 C2PSA, YOLO26's equivalent), these requant boundaries dominate runtime. YOLOv8 has no attention block, which is why it was the only INT8 winner before the fix.

### Performance impact of the fix (benchmark_app, throughput hint, 15s runs)

| Model | Toolchain INT8 FPS | Fixed INT8 FPS | FP32 on NPU FPS | Fixed vs. Toolchain | Fixed vs. FP32 |
|-------|--------------------|----------------|-----------------|---------------------|----------------|
| yolo11n | 185 | 515 | 349 | 2.78× | 1.48× |
| yolo11s | 110 | 270 | 157 | 2.46× | 1.72× |
| yolo26s | 72 | 122 | 113 | 1.70× | 1.08× |
| yolo26m | 41 | 77 | 48 | 1.89× | 1.62× |

Accuracy is preserved: `yolo11n` fixed INT8 evaluates to mAP50=0.644, mAP50-95=0.440 on coco128 (Ultralytics val), within measurement noise of the FP32 baseline.

### Fix applied (see commit)

- `quantization.py`: added `target_device=NPU`, `ignored_scope` for `/attn/`, `/dfl/`, `MatMul`, `fast_bias_correction=True`
- `config.py` + `tests/conftest.py`: default preset switched from `"mixed"` to `"performance"` (symmetric per-channel weights required for VPUX full fusion)

---

## 4. Quantization best practices for NPU

1. Always set `target_device=nncf.TargetDevice.NPU` — the default `ANY` target produces an op set the NPU compiler cannot fully fuse.
2. Use `IgnoredScope` to keep attention, Softmax, and non-fusable MatMul subgraphs in FP16. Full quantization of those is typically a net loss on NPU.
3. Use real, representative calibration data (128–300 samples). Random noise produces miscalibrated scales.
4. Use symmetric per-channel weights via the `PERFORMANCE` preset. Asymmetric or group-wise quantization triggers VPUX compiler fallback paths.
5. After quantization, validate with:
   ```
   benchmark_app -m model.xml -d NPU -hint throughput -report_type detailed_counters
   ```
   and inspect the IR to confirm attention/DFL ops are kept in float precision while the conv backbone is INT8.

---

## 5. Performance settings — open notes

- Manual perf settings are worth exploring, though automatic hints land within 5–10% of optimal for most workloads.
- **NPU turbo:** Set `npu_turbo=1` in the runtime config to inject `NPU_TURBO=YES` at compile time (driver-permitting). This is the biggest single manual lever on NPU — raises the clock to maximum sustained frequency. Warning is logged if set on a non-NPU device.
- **iGPU throughput mode:** 2–4 streams with `num_requests = 2 × streams` is a typical sweet spot.
- **Latency mode** rarely benefits from manual overrides.

---

## 6. Dynamic batching on NPU

NPU requires static input and output shapes baked into the compiled graph — dynamic batching is not supported on this generation. The correct approach is to bake batch into the model at export time (e.g., `dynamic=False, batch=N` in Ultralytics export), which is what the `_b4` / `_b2` model variants already do.

GPU and CPU support dynamic shapes, but each new shape triggers a graph recompile — rarely the right tradeoff for throughput workloads.
