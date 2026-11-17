#!/usr/bin/env python3
"""Annotation tracking management script for AutoClaim-AI Pilot Stage 2.

Usage:
    python ai/scripts/manage_annotation_tracking.py --summary
    python ai/scripts/manage_annotation_tracking.py --list --status REVIEW_REQUIRED
    python ai/scripts/manage_annotation_tracking.py --update --id 1 --status APPROVED --reviewer "SeniorAudit"
    python ai/scripts/manage_annotation_tracking.py --sync
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = WORKSPACE_ROOT / "data" / "parts_pilot"
TRACKING_FILE = PILOT_DIR / "annotation_tracking.csv"
RAW_ANN_DIR = PILOT_DIR / "annotations_raw"
REVIEWED_ANN_DIR = PILOT_DIR / "annotations_reviewed"

VALID_STATUSES = [
    "NOT_STARTED",
    "PRE_LABELED",
    "IN_PROGRESS",
    "ANNOTATED",
    "REVIEW_REQUIRED",
    "APPROVED",
    "REJECTED"
]


def load_tracking():
    if not TRACKING_FILE.exists():
        print(f"[LỖI] Không tìm thấy file tracking: {TRACKING_FILE}")
        sys.exit(1)
    df = pd.read_csv(TRACKING_FILE, dtype={"annotator": str, "reviewer": str, "classes_present": str, "review_notes": str})
    for col in ["annotator", "reviewer", "classes_present", "review_notes"]:
        df[col] = df[col].fillna("")
    return df


def save_tracking(df):
    df.to_csv(TRACKING_FILE, index=False, encoding="utf-8")


def print_summary(df):
    print("=" * 60)
    print("      TIẾN ĐỘ GÁN NHÃN VÀ KIỂM DUYỆT (ANNOTATION TRACKING)")
    print("=" * 60)
    print(f"Tổng số ảnh: {len(df)}")
    
    status_counts = df["status"].value_counts()
    for st in VALID_STATUSES:
        cnt = status_counts.get(st, 0)
        pct = (cnt / len(df)) * 100
        print(f"  {st:20s}: {cnt:2d} ảnh ({pct:5.1f}%)")
        
    print("-" * 60)
    print("Theo phân tập:")
    print(pd.crosstab(df["split"], df["status"]))
    print("=" * 60)


def sync_tracking(df):
    """Quét các file JSON trong annotations_raw và annotations_reviewed để tự động cập nhật số lượng polygon."""
    updated_count = 0
    for idx, r in df.iterrows():
        stem = Path(r["filename"]).stem
        json_raw = RAW_ANN_DIR / f"{stem}.json"
        json_reviewed = REVIEWED_ANN_DIR / f"{stem}.json"
        
        target_json = json_reviewed if json_reviewed.exists() else (json_raw if json_raw.exists() else None)
        
        if target_json:
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
                classes = sorted(list(set(s.get("label", "") for s in shapes if s.get("label"))))
                
                df.loc[idx, "num_polygons"] = len(shapes)
                df.loc[idx, "classes_present"] = ", ".join(classes)
                
                if df.loc[idx, "status"] == "NOT_STARTED" and len(shapes) > 0:
                    if "AI" in str(df.loc[idx, "annotator"]):
                        df.loc[idx, "status"] = "PRE_LABELED"
                    else:
                        df.loc[idx, "status"] = "ANNOTATED"
                    df.loc[idx, "updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated_count += 1
            except Exception as e:
                print(f"Lỗi đọc {target_json}: {e}")
                
    save_tracking(df)
    print(f"[OK] Đã đồng bộ thông tin {updated_count} tệp JSON vào tracking.")


def main():
    parser = argparse.ArgumentParser(description="Quản lý tiến độ gán nhãn Human Annotation")
    parser.add_argument("--summary", action="store_true", help="Hiển thị tổng hợp tiến độ")
    parser.add_argument("--sync", action="store_true", help="Đồng bộ tự động từ thư mục JSON")
    parser.add_argument("--list", action="store_true", help="Liệt kê danh sách ảnh")
    parser.add_argument("--status", choices=VALID_STATUSES, help="Lọc theo trạng thái")
    parser.add_argument("--update", action="store_true", help="Cập nhật trạng thái bản ghi")
    parser.add_argument("--id", type=int, help="Pilot ID cần cập nhật")
    parser.add_argument("--new-status", choices=VALID_STATUSES, help="Trạng thái mới")
    parser.add_argument("--annotator", type=str, help="Tên người gán nhãn")
    parser.add_argument("--reviewer", type=str, help="Tên người kiểm duyệt")
    parser.add_argument("--notes", type=str, help="Ghi chú thẩm định")
    args = parser.parse_args()

    df = load_tracking()

    if args.sync:
        sync_tracking(df)
        df = load_tracking()

    if args.update:
        if args.id is None or args.new_status is None:
            print("[LỖI] Cần truyền --id và --new-status khi cập nhật.")
            sys.exit(1)
        if args.id not in df["pilot_id"].values:
            print(f"[LỖI] Không tìm thấy pilot_id: {args.id}")
            sys.exit(1)
            
        row_idx = df[df["pilot_id"] == args.id].index[0]
        df.loc[row_idx, "status"] = args.new_status
        if args.annotator: df.loc[row_idx, "annotator"] = args.annotator
        if args.reviewer: df.loc[row_idx, "reviewer"] = args.reviewer
        if args.notes: df.loc[row_idx, "review_notes"] = args.notes
        df.loc[row_idx, "updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_tracking(df)
        print(f"[OK] Đã cập nhật ảnh #{args.id} ({df.loc[row_idx, 'filename']}) -> {args.new_status}")
        return

    if args.list:
        sub_df = df
        if args.status:
            sub_df = df[df["status"] == args.status]
        print(sub_df[["pilot_id", "filename", "split", "viewpoint", "status", "num_polygons", "review_notes"]].to_string())
        return

    # Mặc định in summary
    print_summary(df)


if __name__ == "__main__":
    main()
