"""Find the first node where a QDQ INT8 graph diverges from its FP32 original.

Why this exists
---------------
``results/q035_detector_confidence.csv`` records that the INT8 detector emits a
confidence of exactly 0.0 on 200/200 test images while its box coordinates
survive intact. "Exactly zero" is not degradation, so the interesting question is
not how much accuracy was lost but *which node destroys the signal*. Attributing
the collapse to an architectural property (SiLU tails, depthwise ranges) without
finding that node would be speculation.

Method
------
Both graphs are re-run with every intermediate tensor promoted to a graph output
and with ORT's graph optimizations disabled, so nodes are not fused away. Tensors
are matched by name: ONNX Runtime's quantizer preserves the original tensor name
and appends ``_DequantizeLinear_Output`` to the dequantized copy, so an FP32
tensor ``T`` is compared against whichever of ``T`` or
``T_DequantizeLinear_Output`` the INT8 graph actually carries.

The report walks the INT8 graph in topological order and flags the first tensor
that is entirely zero in INT8 while non-zero in FP32. Only summary statistics are
retained per tensor -- a YOLO graph at 640x640 holds far too much activation to
keep in memory all at once, which is also why tensors are exposed in chunks.

Usage::

    python scripts/diagnose_int8_divergence.py \\
        --fp32 results/int8_models/iteration4_ultra.onnx \\
        --int8 results/int8_models/iteration4_ultra_int8.onnx
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from src.results import RESULTS_DIR
from src.utils import configure_logging

from evaluate_common import ground_truth_labels, list_test_images  # noqa: E402

logger = logging.getLogger("diagnose_int8_divergence")

DIVERGENCE_FIELDS = [
    "order", "tensor", "producer_node", "op_type",
    "fp32_absmax", "int8_absmax", "fp32_all_zero", "int8_all_zero",
    "diverged_to_zero", "scale", "zero_point",
]


def letterboxed(path: Path, size: int = 640) -> np.ndarray:
    """Ultralytics-style preprocessing, matching evaluate_onnx_detseg."""
    from ultralytics.data.augment import LetterBox

    image = LetterBox((size, size), auto=False)(image=cv2.imread(str(path)))
    chw = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def tensor_stats(model_path: Path, names: list[str], feed: np.ndarray, chunk: int = 120):
    """
    Absmax and all-zero flags for named tensors, computed in chunks.

    Promoting every tensor to an output at once exhausts memory on a detector
    graph, so the model is re-run per chunk. Optimizations are disabled because a
    fused node has no observable intermediate output, and the fusion boundaries
    differ between the FP32 and INT8 graphs.
    """
    base = onnx.load(str(model_path))
    stats: dict[str, tuple[float, bool]] = {}

    for start in range(0, len(names), chunk):
        batch = names[start : start + chunk]
        model = onnx.ModelProto()
        model.CopyFrom(base)
        existing = {o.name for o in model.graph.output}
        for name in batch:
            if name not in existing:
                model.graph.output.append(onnx.helper.ValueInfoProto(name=name))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk %d-%d unusable: %s", start, start + len(batch), exc)
            continue

        out_names = [o.name for o in session.get_outputs()]
        values = session.run(out_names, {session.get_inputs()[0].name: feed})
        for name, value in zip(out_names, values):
            if name in batch:
                array = np.asarray(value, dtype=np.float64)
                stats[name] = (float(np.abs(array).max()) if array.size else 0.0,
                               bool(array.size and np.all(array == 0)))
        logger.info("  probed %d/%d tensors", min(start + chunk, len(names)), len(names))

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate the first INT8 divergence.")
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None,
                        help="Defaults to the first test image containing fire or smoke.")
    parser.add_argument("--output", default="q035_int8_divergence.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.image is None:
        paths = list_test_images("test")
        truth = ground_truth_labels(paths, "test")
        args.image = paths[next(i for i, t in enumerate(truth) if t != 0)]
    logger.info("Probing on %s", args.image.name)
    feed = letterboxed(args.image)

    int8_model = onnx.load(str(args.int8))
    fp32_model = onnx.load(str(args.fp32))
    initializers = {i.name: numpy_helper.to_array(i) for i in int8_model.graph.initializer}

    # Quantization parameters, keyed by the tensor each QDQ node produces.
    qparams: dict[str, tuple[float, float]] = {}
    for node in int8_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) >= 3:
            scale = initializers.get(node.input[1])
            zero = initializers.get(node.input[2])
            if scale is not None and scale.size == 1:
                qparams[node.output[0]] = (float(scale.reshape(-1)[0]),
                                           float(np.asarray(zero).reshape(-1)[0]) if zero is not None else 0.0)

    fp32_tensors = [o for n in fp32_model.graph.node for o in n.output]
    fp32_available = set(fp32_tensors)

    # Walk the INT8 graph in topological (node) order.
    probes: list[tuple[str, str, str, str]] = []   # int8 name, fp32 name, node name, op type
    for node in int8_model.graph.node:
        for out in node.output:
            base = out
            for suffix in ("_DequantizeLinear_Output", "_QuantizeLinear_Output"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            if base in fp32_available:
                probes.append((out, base, node.name, node.op_type))

    logger.info("Matched %d tensors between the two graphs", len(probes))

    logger.info("Probing INT8 graph...")
    int8_stats = tensor_stats(args.int8, [p[0] for p in probes], feed)
    logger.info("Probing FP32 graph...")
    fp32_stats = tensor_stats(args.fp32, sorted({p[1] for p in probes}), feed)

    rows = []
    first = None
    for order, (int8_name, fp32_name, node_name, op_type) in enumerate(probes):
        if int8_name not in int8_stats or fp32_name not in fp32_stats:
            continue
        i_max, i_zero = int8_stats[int8_name]
        f_max, f_zero = fp32_stats[fp32_name]
        diverged = bool(i_zero and not f_zero)
        scale, zero_point = qparams.get(int8_name, ("", ""))
        rows.append({
            "order": order, "tensor": int8_name, "producer_node": node_name,
            "op_type": op_type, "fp32_absmax": f_max, "int8_absmax": i_max,
            "fp32_all_zero": int(f_zero), "int8_all_zero": int(i_zero),
            "diverged_to_zero": int(diverged), "scale": scale, "zero_point": zero_point,
        })
        if diverged and first is None:
            first = rows[-1]

    out_path = RESULTS_DIR / args.output
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIVERGENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d comparisons to %s", len(rows), out_path)

    print(f"\nCompared {len(rows)} tensors present in both graphs.")
    zeroed = [r for r in rows if r["diverged_to_zero"]]
    if not zeroed:
        print("No tensor is all-zero in INT8 while non-zero in FP32.")
        return

    print(f"{len(zeroed)} tensor(s) go to exactly zero in INT8 but not in FP32.")
    print("\nFIRST divergence in topological order:")
    for key in ("order", "tensor", "producer_node", "op_type",
                "fp32_absmax", "int8_absmax", "scale", "zero_point"):
        print(f"  {key:16s} {first[key]}")


if __name__ == "__main__":
    main()
