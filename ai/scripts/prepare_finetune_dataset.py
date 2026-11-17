#!/usr/bin/env python3
"""AutoClaim-AI: Consolidate Fine-Tuning Dataset (Domain-Adaptive Fine-Tuning).

Consolidates three data sources into `ai/data/parts_finetune_8cls/`:
1. Human-reviewed Pilot images (status in ['ANNOTATED', 'APPROVED'] from annotation_tracking.csv).
2. Auxiliary Fender dataset (`ai/data/parts_fender_aux/`) with high-resolution ve_che_banh polygons.
3. 400 Replay images sampled from `ai/data/parts_carparts_8cls/` to prevent catastrophic forgetting.

Generates `ai/configs/data_parts_finetune_8cls.yaml`.
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = WORKSPACE_ROOT / "data" / "parts_pilot"
TRACKING_FILE = PILOT_DIR / "annotation_tracking.csv"
PILOT_RAW_DIR = PILOT_DIR / "annotations_raw"
PILOT_REV_DIR = PILOT_DIR / "annotations_reviewed"
PILOT_IMG_DIR = PILOT_DIR / "images"

FENDER_AUX_DIR = WORKSPACE_ROOT / "data" / "parts_fender_aux"
CARPARTS_DIR = WORKSPACE_ROOT / "data" / "parts_carparts_8cls"
DEST_DIR = WORKSPACE_ROOT / "data" / "parts_finetune_8cls"
CONFIG_FILE = WORKSPACE_ROOT / "configs" / "data_parts_finetune_8cls.yaml"

CLASS_NAMES = [
    "can_truoc",
    "can_sau",
    "nap_capo",
    "ve_che_banh",
    "cua_truoc",
    "cua_sau",
    "kinh_xe",
    "den_xe",
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_labelme_json(json_path, img_w, img_h):
    """Convert LabelMe JSON to YOLO polygon lines."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w = data.get("imageWidth") or img_w
    h = data.get("imageHeight") or img_h
    if not w or not h:
        return []

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if label not in CLASS_MAP:
            continue
        cid = CLASS_MAP[label]
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        coords = []
        for pt in points:
            x_norm = max(0.0, min(1.0, pt[0] / float(w)))
            y_norm = max(0.0, min(1.0, pt[1] / float(h)))
            coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        lines.append(f"{cid} " + " ".join(coords))
    return lines


def copy_or_link(src: Path, dst: Path):
    """Copy or hardlink file safely."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def count_labels_in_file(lbl_path, class_counter):
    if not lbl_path.exists():
        return
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                if 0 <= cid < len(CLASS_NAMES):
                    class_counter[cid] += 1
            except ValueError:
                pass


def main():
    print("=" * 65)
    print("   AUTOCLAIM-AI: CONSOLIDATE FINE-TUNING DATASET (STAGE 2 STEP 4)")
    print("=" * 65)

    # Recreate destination directory structure
    for split in ["train", "val"]:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)

    # -------------------------------------------------------------
    # 1. PROCESS PILOT DATASET (Human Annotated / Approved)
    # -------------------------------------------------------------
    print("\n[1/3] Quét dữ liệu Pilot Dataset...")
    pilot_reviewed_count = 0
    if TRACKING_FILE.exists():
        df_track = pd.read_csv(TRACKING_FILE)
        # Find approved / annotated
        reviewed_df = df_track[df_track["status"].isin(["ANNOTATED", "APPROVED"])]
        print(f"  Tìm thấy {len(reviewed_df)} ảnh Pilot đã kiểm duyệt (ANNOTATED / APPROVED).")

        for _, row in reviewed_df.iterrows():
            fn = row["filename"]
            stem = Path(fn).stem
            split = str(row.get("split", "train")).lower()
            if split not in ["train", "val"]:
                split = "train"

            # Check reviewed first, then raw
            json_path = PILOT_REV_DIR / f"{stem}.json"
            if not json_path.exists():
                json_path = PILOT_RAW_DIR / f"{stem}.json"

            if not json_path.exists():
                print(f"  [Bỏ qua] Không tìm thấy file JSON cho {fn}")
                continue

            # Image path
            img_path = PILOT_IMG_DIR / fn
            if not img_path.exists():
                img_path = Path(str(row.get("image_path", "")))
            if not img_path.exists():
                print(f"  [Bỏ qua] Không tìm thấy ảnh nguồn cho {fn}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            yolo_lines = parse_labelme_json(json_path, img_w, img_h)
            if not yolo_lines:
                print(f"  [Cảnh báo] File {json_path.name} không chứa polygon hợp lệ.")
                continue

            dst_img_name = f"pilot_{fn}"
            dst_lbl_name = f"pilot_{stem}.txt"

            dst_img = DEST_DIR / "images" / split / dst_img_name
            dst_lbl = DEST_DIR / "labels" / split / dst_lbl_name

            copy_or_link(img_path, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            target_counts = train_class_counts if split == "train" else val_class_counts
            count_labels_in_file(dst_lbl, target_counts)
            pilot_reviewed_count += 1
            print(f"  -> Nhập Pilot: {dst_img_name} ({split}) - {len(yolo_lines)} polygons")
    else:
        print("  [Cảnh báo] Không tìm thấy annotation_tracking.csv")

    # -------------------------------------------------------------
    # 2. PROCESS AUXILIARY FENDER DATASET
    # -------------------------------------------------------------
    print("\n[2/3] Quét dữ liệu bổ trợ Ve Che Banh (parts_fender_aux)...")
    fender_train_count = 0
    fender_val_count = 0
    if FENDER_AUX_DIR.exists():
        fender_imgs = sorted(list((FENDER_AUX_DIR / "images").glob("*.*")))
        # Filter valid images that have matching labels
        valid_pairs = []
        for img_p in fender_imgs:
            lbl_p = FENDER_AUX_DIR / "labels" / f"{img_p.stem}.txt"
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                valid_pairs.append((img_p, lbl_p))

        print(f"  Tìm thấy {len(valid_pairs)} cặp ảnh/nhãn Ve che banh hợp lệ.")
        rng = random.Random(42)
        rng.shuffle(valid_pairs)

        # 80/20 train/val split
        val_size = int(len(valid_pairs) * 0.2)
        val_pairs = valid_pairs[:val_size]
        train_pairs = valid_pairs[val_size:]

        for img_p, lbl_p in train_pairs:
            safe_stem = img_p.stem.replace(" ", "_")
            dst_img = DEST_DIR / "images" / "train" / f"fender_{safe_stem}{img_p.suffix.lower()}"
            dst_lbl = DEST_DIR / "labels" / "train" / f"fender_{safe_stem}.txt"
            copy_or_link(img_p, dst_img)
            copy_or_link(lbl_p, dst_lbl)
            count_labels_in_file(dst_lbl, train_class_counts)
            fender_train_count += 1

        for img_p, lbl_p in val_pairs:
            safe_stem = img_p.stem.replace(" ", "_")
            dst_img = DEST_DIR / "images" / "val" / f"fender_{safe_stem}{img_p.suffix.lower()}"
            dst_lbl = DEST_DIR / "labels" / "val" / f"fender_{safe_stem}.txt"
            copy_or_link(img_p, dst_img)
            copy_or_link(lbl_p, dst_lbl)
            count_labels_in_file(dst_lbl, val_class_counts)
            fender_val_count += 1

        print(f"  -> Nhập Fender: {fender_train_count} train, {fender_val_count} val.")
    else:
        print(f"  [Lỗi] Không tìm thấy thư mục {FENDER_AUX_DIR}")

    # -------------------------------------------------------------
    # 3. SAMPLE REPLAY DATASET FROM CARPARTS (Prevent Catastrophic Forgetting)
    # -------------------------------------------------------------
    print("\n[3/3] Trích mẫu 400 ảnh Replay từ carparts_8cls/images/train...")
    replay_count = 0
    carparts_train_img_dir = CARPARTS_DIR / "images" / "train"
    carparts_train_lbl_dir = CARPARTS_DIR / "labels" / "train"

    if carparts_train_img_dir.exists():
        all_carparts_imgs = sorted(list(carparts_train_img_dir.glob("*.*")))
        valid_carparts = []
        for img_p in all_carparts_imgs:
            lbl_p = carparts_train_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                valid_carparts.append((img_p, lbl_p))

        rng = random.Random(42)
        sampled_carparts = rng.sample(valid_carparts, min(400, len(valid_carparts)))

        for img_p, lbl_p in sampled_carparts:
            safe_stem = img_p.stem.replace(" ", "_")
            dst_img = DEST_DIR / "images" / "train" / f"replay_{safe_stem}{img_p.suffix.lower()}"
            dst_lbl = DEST_DIR / "labels" / "train" / f"replay_{safe_stem}.txt"
            copy_or_link(img_p, dst_img)
            copy_or_link(lbl_p, dst_lbl)
            count_labels_in_file(dst_lbl, train_class_counts)
            replay_count += 1

        print(f"  -> Nhập Replay: {replay_count} ảnh vào tập train.")
    else:
        print(f"  [Lỗi] Không tìm thấy {carparts_train_img_dir}")

    # -------------------------------------------------------------
    # 4. GENERATE YAML CONFIG
    # -------------------------------------------------------------
    config_content = f"""# Cấu hình tinh chỉnh YOLOv8-seg (Domain-Adaptive Fine-Tuning) cho AutoClaim-AI
# Dữ liệu hợp nhất: Pilot (Human-reviewed) + Fender Aux (ve_che_banh) + Replay CarParts
path: {DEST_DIR.as_posix()}
train: images/train
val: images/val

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

# Siêu tham số tăng cường nhẹ cho fine-tuning để giữ chi tiết vùng tổn thương/cận cảnh
mixup: 0.1
mosaic: 0.5
scale: 0.3
degrees: 5.0
"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"\n[Cấu hình] Đã tạo file YAML: {CONFIG_FILE}")

    # -------------------------------------------------------------
    # 5. SUMMARY REPORT
    # -------------------------------------------------------------
    total_train_imgs = len(list((DEST_DIR / "images" / "train").glob("*.*")))
    total_val_imgs = len(list((DEST_DIR / "images" / "val").glob("*.*")))

    print("\n" + "=" * 65)
    print("      TỔNG HỢP BỘ DỮ LIỆU TINH CHỈNH (PARTS_FINETUNE_8CLS)")
    print("=" * 65)
    print(f"Tập Train: {total_train_imgs} ảnh (Pilot: {pilot_reviewed_count if pilot_reviewed_count else 0}, Fender: {fender_train_count}, Replay: {replay_count})")
    print(f"Tập Val  : {total_val_imgs} ảnh (Fender: {fender_val_count})")
    print("-" * 65)
    print(f"{'Class ID':<10} {'Tên Lớp (Class Name)':<20} {'Train Polygons':<15} {'Val Polygons':<15}")
    print("-" * 65)
    for cid in range(8):
        name = CLASS_NAMES[cid]
        tr_c = train_class_counts[cid]
        val_c = val_class_counts[cid]
        print(f"{cid:<10} {name:<20} {tr_c:<15} {val_c:<15}")
    print("=" * 65)
    print(f"Tổng số polygon: Train = {sum(train_class_counts.values())}, Val = {sum(val_class_counts.values())}")
    print("Dataset đã sẵn sàng tại:", DEST_DIR)


if __name__ == "__main__":
    main()
