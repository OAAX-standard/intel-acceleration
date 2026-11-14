#!/usr/bin/env python3
"""Build the tiny model `robustness_test` runs: Reshape(x[1,4] f32, shape[2] i64) as an OpenVINO IR zip.

Why this model:
- a shape that does not hold 4 elements makes the inference itself fail (on any device), which is how a
  request fails on real hardware (e.g. an NPU that rejects a model at run time);
- its output is dynamic and has a 1140-character name, like detectors with the class list in the output name.

The IR is written by hand (it has no weights), so this needs nothing but the standard library and runs
wherever the tests run.

Usage: python make_robustness_model.py OUT.zip
"""

import sys
import zipfile
from pathlib import Path

LONG_OUTPUT_NAME = "bboxes-format:xyxysc;" + ";".join(f"{i}:class-number-{i}" for i in range(60))

IR_XML = f"""<?xml version="1.0"?>
<net name="robustness" version="11">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data shape="1,4" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="x">
					<dim>1</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="shape" type="Parameter" version="opset1">
			<data shape="2" element_type="i64" />
			<output>
				<port id="0" precision="I64" names="shape">
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Reshape_3" type="Reshape" version="opset1">
			<data special_zero="false" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
				</port>
				<port id="1" precision="I64">
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="{LONG_OUTPUT_NAME}">
					<dim>-1</dim>
					<dim>-1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Result_4" type="Result" version="opset1" output_names="{LONG_OUTPUT_NAME}">
			<input>
				<port id="0" precision="FP32">
					<dim>-1</dim>
					<dim>-1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
	</edges>
	<rt_info />
</net>
"""


def build(out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as z:
        z.writestr("model.xml", IR_XML)
        z.writestr("model.bin", b"")  # the model has no constants


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    assert len(LONG_OUTPUT_NAME) > 256
    build(Path(sys.argv[1]))
    print(f"wrote {sys.argv[1]} (output name: {len(LONG_OUTPUT_NAME)} characters)")
