#!/usr/bin/env python3
"""Build the tiny model `robustness_test` runs: Reshape(x[1,4] f32, shape[2] i64) as an OpenVINO IR zip.

Why this model:
- a shape that does not hold 4 elements makes the inference itself fail (on any device), which is how a
  request fails on real hardware (e.g. an NPU that rejects a model at run time);
- its output is dynamic and has a 900-character name, like detectors with the class list in the output name.

Usage: python make_robustness_model.py OUT.zip
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

LONG_OUTPUT_NAME = "bboxes-format:xyxysc;" + ";".join(f"{i}:class-number-{i}" for i in range(60))


def build(out_zip: Path) -> None:
    x = ops.parameter([1, 4], np.float32, name="x")
    x.output(0).set_names({"x"})
    shape = ops.parameter([2], np.int64, name="shape")
    shape.output(0).set_names({"shape"})
    reshaped = ops.reshape(x, shape, special_zero=False)
    reshaped.output(0).set_names({LONG_OUTPUT_NAME})
    model = ov.Model([reshaped], [x, shape], "robustness")
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "model.xml"
        ov.save_model(model, xml, compress_to_fp16=False)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w") as z:
            z.write(xml, "model.xml")
            z.write(xml.with_suffix(".bin"), "model.bin")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    assert len(LONG_OUTPUT_NAME) > 256
    build(Path(sys.argv[1]))
    print(f"wrote {sys.argv[1]} (output name: {len(LONG_OUTPUT_NAME)} characters)")
