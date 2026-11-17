#!/usr/bin/env python3
"""Launcher script for AutoClaim-AI Human Annotation Tool (Stage 2 Pilot).

Supports launching LabelMe with pre-configured 8 vehicle part classes:
0: can_truoc
1: can_sau
2: nap_capo
3: ve_che_banh
4: cua_truoc
5: cua_sau
6: kinh_xe
7: den_xe
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = WORKSPACE_ROOT / "data" / "parts_pilot"
IMAGES_DIR = PILOT_DIR / "images"
LABELS_FILE = PILOT_DIR / "annotation_config" / "labels.txt"
OUTPUT_DIR = PILOT_DIR / "annotations_raw"


def check_prerequisites():
    """Kiểm tra sự tồn tại của thư mục ảnh và cấu hình nhãn."""
    if not IMAGES_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục ảnh Pilot tại: {IMAGES_DIR}")
        sys.exit(1)
        
    num_images = len([f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if num_images != 80:
        print(f"[CẢNH BÁO] Số ảnh trong {IMAGES_DIR} là {num_images} (kỳ vọng đúng 80 ảnh).")
    else:
        print(f"[OK] Đã xác thực 80 ảnh Pilot tại: {IMAGES_DIR}")

    if not LABELS_FILE.exists():
        print(f"[LỖI] Không tìm thấy file nhãn tại: {LABELS_FILE}")
        sys.exit(1)
    print(f"[OK] File nhãn chuẩn 8 lớp: {LABELS_FILE}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thư mục lưu nhãn thô: {OUTPUT_DIR}")


def check_labelme_installed():
    """Kiểm tra công cụ LabelMe trong môi trường Python hoặc hệ thống."""
    # Kiểm tra lệnh labelme trên PATH
    which_labelme = shutil.which("labelme")
    if which_labelme:
        return ["labelme"]
        
    # Kiểm tra qua module python
    try:
        import labelme
        return [sys.executable, "-m", "labelme"]
    except ImportError:
        pass
        
    return None


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy công cụ gán nhãn Human Annotation cho AutoClaim-AI")
    parser.add_argument("--tool", choices=["labelme", "anylabeling"], default="labelme", help="Công cụ gán nhãn")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Thư mục xuất file JSON")
    args = parser.parse_args()

    print("=" * 65)
    print("      AUTOCLAIM-AI — HUMAN ANNOTATION TOOL LAUNCHER")
    print("=" * 65)
    
    check_prerequisites()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_base = check_labelme_installed()
    
    if cmd_base is None:
        print("\n" + "!" * 65)
        print("[THÔNG BÁO] Chưa cài đặt thư viện 'labelme' trong môi trường Python.")
        print("!" * 65)
        print("\nĐể cài đặt và chạy LabelMe trên Windows, vui lòng thực thi lệnh sau:")
        print("   pip install pyqt6 labelme")
        print("\nHoặc tải bản Portable LabelMe.exe trực tiếp từ GitHub Releases:")
        print("   https://github.com/labelmeai/labelme/releases")
        print("\nSau đó chạy lại script này hoặc mở LabelMe với tham số chuẩn:")
        print(f"   labelme {IMAGES_DIR} --labels {LABELS_FILE} --output {out_dir} --nodata")
        print("=" * 65)
        sys.exit(0)

    # Cấu hình tham số khởi chạy LabelMe chuẩn hóa
    # --nodata: Không nhúng base64 ảnh vào JSON (giúp file nhẹ < 10KB)
    # --labels: Khóa danh sách nhãn đúng 8 lớp
    # --output: Lưu JSON trực tiếp vào annotations_raw
    cmd = cmd_base + [
        str(IMAGES_DIR),
        "--labels", str(LABELS_FILE),
        "--output", str(out_dir),
        "--nodata",
        "--autosave"
    ]

    print("\n[KHỞI CHẠY] Đang mở LabelMe...")
    print(f"Lệnh thực thi: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"[LỖI] Không thể khởi chạy LabelMe: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
