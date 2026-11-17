"""Xuất mô hình YOLO PyTorch (.pt) sang định dạng ONNX (.onnx) tương thích onnxruntime.

Cách dùng:
    python ai/scripts/export_onnx.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO


DEFAULT_MODELS = [
    Path("ai/models/weights/damage_best.pt"),
    Path("ai/models/weights/parts_seg.pt"),
]


def export_model_to_onnx(
    pt_path: Path,
    *,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> Path:
    """Xuất 1 mô hình Ultralytics .pt sang .onnx và kiểm tra tương thích onnxruntime."""
    if not pt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {pt_path}")

    print(f"\n>>> Đang xuất {pt_path} -> ONNX (opset={opset}, imgsz={imgsz})...")
    model = YOLO(str(pt_path))
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
    )
    onnx_path = Path(exported_path_str)

    # Kiểm tra nhanh bằng onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]
    outputs = [o.name for o in sess.get_outputs()]
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    [OK] Đã xuất: {onnx_path} ({size_mb:.2f} MB)")
    print(f"    [ORT] Inputs: {inputs} | Outputs: {outputs}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AutoClaim-AI YOLO weights to ONNX")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Tắt onnxsim simplify")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    exported: list[Path] = []
    for rel_pt in DEFAULT_MODELS:
        pt_path = root / rel_pt
        onnx_path = export_model_to_onnx(
            pt_path,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=not args.no_simplify,
        )
        exported.append(onnx_path)

    print("\n=== TỔNG KẾT XUẤT ONNX ===")
    for p in exported:
        size_mb = p.stat().st_size / (1024 * 1024)
        flag = " (> 90 MB!)" if size_mb > 90 else ""
        print(f"  - {p.relative_to(root)}: {size_mb:.2f} MB{flag}")


if __name__ == "__main__":
    main()
