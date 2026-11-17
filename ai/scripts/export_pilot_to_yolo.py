#!/usr/bin/env python3
"""Export verified LabelMe annotations to YOLOv8-seg format.

Constraints:
1. ONLY export images with status == 'APPROVED' (unless --allow-unapproved flag is set for test dry-run).
2. Preserves strict 8-class IDs:
    0: can_truoc
    1: can_sau
    2: nap_capo
    3: ve_che_banh
    4: cua_truoc
    5: cua_sau
    6: kinh_xe
    7: den_xe
3. Normalized coordinates in [0.0, 1.0] with 6 decimal precision.
4. Strictly isolated under ai/data/parts_pilot/annotation_exports/ (no overwriting existing datasets).
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = WORKSPACE_ROOT / "data" / "parts_pilot"
MANIFEST_FILE = PILOT_DIR / "pilot_manifest.csv"
TRACKING_FILE = PILOT_DIR / "annotation_tracking.csv"
DEFAULT_ANN_DIR = PILOT_DIR / "annotations_reviewed"
DEFAULT_EXPORT_DIR = PILOT_DIR / "annotation_exports"

CLASS_MAP = {
    "can_truoc": 0,
    "can_sau": 1,
    "nap_capo": 2,
    "ve_che_banh": 3,
    "cua_truoc": 4,
    "cua_sau": 5,
    "kinh_xe": 6,
    "den_xe": 7,
}


def convert_labelme_to_yolo_lines(json_path, img_w, img_h):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use dimension from JSON if available and positive
    w = data.get("imageWidth") or img_w
    h = data.get("imageHeight") or img_h
    if not w or not h:
        return []

    lines = []
    for s in data.get("shapes", []):
        label = s.get("label", "").strip()
        if label not in CLASS_MAP:
            continue
        cid = CLASS_MAP[label]
        points = s.get("points", [])
        if len(points) < 3:
            continue

        norm_coords = []
        for pt in points:
            x_norm = max(0.0, min(1.0, pt[0] / float(w)))
            y_norm = max(0.0, min(1.0, pt[1] / float(h)))
            norm_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        line = f"{cid} " + " ".join(norm_coords)
        lines.append(line)

    return lines


def export_dataset(ann_dir, export_dir, allow_unapproved=False):
    print("=" * 65)
    print("      AUTOCLAIM-AI — YOLO SEGMENTATION EXPORTER")
    print("=" * 65)

    if not MANIFEST_FILE.exists() or not TRACKING_FILE.exists():
        print("[LỖI] Thiếu tệp manifest hoặc tracking.")
        sys.exit(1)

    df_manifest = pd.read_csv(MANIFEST_FILE)
    df_tracking = pd.read_csv(TRACKING_FILE)

    ann_dir = Path(ann_dir)
    export_dir = Path(export_dir)

    train_img_dir = export_dir / "images" / "train"
    val_img_dir = export_dir / "images" / "val"
    train_lbl_dir = export_dir / "labels" / "train"
    val_lbl_dir = export_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0
    total_polygons = 0

    for idx, r in df_manifest.iterrows():
        pid = r["pilot_id"]
        fn = r["filename"]
        split = r["split"]
        stem = Path(fn).stem

        track_row = df_tracking[df_tracking["pilot_id"] == pid]
        status = track_row["status"].values[0] if len(track_row) > 0 else "NOT_STARTED"

        if status != "APPROVED" and not allow_unapproved:
            skipped_count += 1
            continue

        json_p = ann_dir / f"{stem}.json"
        if not json_p.exists():
            skipped_count += 1
            continue

        src_img_p = Path(r["image_path"])
        if not src_img_p.exists():
            src_img_p = PILOT_DIR / "images" / fn

        if not src_img_p.exists():
            print(f"[CẢNH BÁO] Không tìm thấy ảnh nguồn: {src_img_p}")
            skipped_count += 1
            continue

        with Image.open(src_img_p) as im:
            img_w, img_h = im.size

        yolo_lines = convert_labelme_to_yolo_lines(json_p, img_w, img_h)
        if not yolo_lines:
            print(f"[CẢNH BÁO] File {json_p.name} không có polygon hợp lệ, bỏ qua.")
            skipped_count += 1
            continue

        # Target paths
        target_img_dir = train_img_dir if split == "train" else val_img_dir
        target_lbl_dir = train_lbl_dir if split == "train" else val_lbl_dir

        dst_img = target_img_dir / fn
        dst_lbl = target_lbl_dir / f"{stem}.txt"

        # Hard link image (or copy if cross-filesystem)
        if not dst_img.exists():
            try:
                os.link(src_img_p, dst_img)
            except Exception:
                shutil.copy2(src_img_p, dst_img)

        # Write YOLO label
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        exported_count += 1
        total_polygons += len(yolo_lines)

    # Write dataset yaml for YOLO training
    yaml_content = f"""# Cấu hình Dataset YOLOv8-seg cho Pilot Stage 2
path: {export_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/val

nc: 8
names:
  0: can_truoc
  1: can_sau
  2: nap_capo
  3: ve_che_banh
  4: cua_truoc
  5: cua_sau
  6: kinh_xe
  7: den_xe
"""
    yaml_path = export_dir / "dataset_parts_pilot.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[KẾT QUẢ XUẤT YOLO]")
    print(f"  - Số ảnh đã xuất thành công: {exported_count}")
    print(f"  - Số ảnh bỏ qua (chưa duyệt / thiếu file): {skipped_count}")
    print(f"  - Tổng số polygon đã chuyển đổi: {total_polygons}")
    print(f"  - Thư mục xuất: {export_dir}")
    print(f"  - File cấu hình: {yaml_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi nhãn LabelMe sang định dạng YOLOv8 Segmentation")
    parser.add_argument("--ann-dir", type=str, default=str(DEFAULT_ANN_DIR), help="Thư mục chứa LabelMe JSON")
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR), help="Thư mục xuất kết quả")
    parser.add_argument("--allow-unapproved", action="store_true", help="Cho phép xuất ảnh chưa APPROVED (chỉ dùng khi test thử nghiệm)")
    args = parser.parse_args()

    export_dataset(args.ann_dir, args.export_dir, args.allow_unapproved)


if __name__ == "__main__":
    main()
