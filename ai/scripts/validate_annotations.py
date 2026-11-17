#!/usr/bin/env python3
"""Quality Control (QC) Validator for AutoClaim-AI Human Annotation.

Validates:
1. Annotation JSON syntax & schema.
2. Source image existence & dimension match.
3. Class validity (strictly 8 target classes).
4. Polygon vertices count (>= 3 points).
5. Coordinates bounds ([0, W], [0, H]).
6. Non-zero polygon area (Shoelace formula).
7. Duplicate / anomalous polygon overlap (IoU >= 0.90).
8. APPROVED images without annotations or with invalid annotations.

Outputs:
    docs/stage2_annotation_quality_report.md
"""

import os
import sys
import json
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
IMAGES_DIR = PILOT_DIR / "images"
DEFAULT_RAW_DIR = PILOT_DIR / "annotations_raw"
DEFAULT_REVIEWED_DIR = PILOT_DIR / "annotations_reviewed"
DEFAULT_REPORT_OUT = WORKSPACE_ROOT.parent / "docs" / "stage2_annotation_quality_report.md"

VALID_CLASSES = [
    "can_truoc",
    "can_sau",
    "nap_capo",
    "ve_che_banh",
    "cua_truoc",
    "cua_sau",
    "kinh_xe",
    "den_xe"
]


def polygon_area(points):
    """Tính diện tích đa giác bằng công thức Shoelace."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_bounds(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def validate_single_json(json_path, image_path):
    issues = []
    shapes_info = []

    # 1. Parse JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [{"type": "INVALID_JSON", "msg": f"Lỗi cú pháp JSON: {e}"}], []

    # 2. Check image existence & dimensions
    if not image_path.exists():
        issues.append({"type": "MISSING_IMAGE", "msg": f"Ảnh gốc không tồn tại: {image_path}"})
        img_w, img_h = data.get("imageWidth", 1000), data.get("imageHeight", 667)
    else:
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            issues.append({"type": "UNREADABLE_IMAGE", "msg": f"Không đọc được ảnh: {e}"})
            img_w, img_h = 1000, 667

    # Check JSON dimension match
    json_w = data.get("imageWidth")
    json_h = data.get("imageHeight")
    if json_w and json_h and (json_w != img_w or json_h != img_h):
        issues.append({"type": "DIMENSION_MISMATCH", "msg": f"Kích thước JSON ({json_w}x{json_h}) không khớp ảnh ({img_w}x{img_h})"})

    # 3. Check shapes
    shapes = data.get("shapes", [])
    if not shapes:
        issues.append({"type": "EMPTY_ANNOTATION", "msg": "Tệp JSON không chứa đa giác nào (0 shapes)"})

    for i, s in enumerate(shapes):
        label = s.get("label", "").strip()
        points = s.get("points", [])
        
        # Class check
        if label not in VALID_CLASSES:
            issues.append({"type": "INVALID_CLASS", "msg": f"Shape #{i}: Class '{label}' không thuộc 8 lớp chuẩn"})

        # Vertex count check
        if len(points) < 3:
            issues.append({"type": "TOO_FEW_POINTS", "msg": f"Shape #{i} ({label}): Có {len(points)} đỉnh (< 3)"})
            continue

        # Coordinate bounds check
        out_of_bounds = False
        for pt in points:
            x, y = pt[0], pt[1]
            if x < 0 or x > img_w or y < 0 or y > img_h:
                out_of_bounds = True
                break
        if out_of_bounds:
            issues.append({"type": "OUT_OF_BOUNDS", "msg": f"Shape #{i} ({label}): Tọa độ vượt ngoài kích thước ảnh ({img_w}x{img_h})"})

        # Area check
        area = polygon_area(points)
        if area <= 10.0:  # Diện tích quá nhỏ hoặc suy biến
            issues.append({"type": "DEGENERATE_POLYGON", "msg": f"Shape #{i} ({label}): Diện tích quá nhỏ hoặc suy biến (Area={area:.1f} px^2)"})

        bbox = polygon_bounds(points)
        shapes_info.append({"idx": i, "label": label, "area": area, "bbox": bbox, "points": points})

    # Check duplicate shapes (IoU > 0.90 cùng class)
    for i in range(len(shapes_info)):
        for j in range(i + 1, len(shapes_info)):
            s1, s2 = shapes_info[i], shapes_info[j]
            if s1["label"] == s2["label"]:
                iou = bbox_iou(s1["bbox"], s2["bbox"])
                if iou >= 0.90:
                    issues.append({"type": "DUPLICATE_ANNOTATION", "msg": f"Nghi vấn trùng lặp: Shape #{s1['idx']} và #{s2['idx']} (class '{s1['label']}', BBox IoU={iou:.2f})"})

    return issues, shapes_info


def run_qc(ann_dir, report_out_path):
    print("=" * 65)
    print("      AUTOCLAIM-AI — ANNOTATION QUALITY CONTROL (QC)")
    print("=" * 65)

    if not MANIFEST_FILE.exists():
        print(f"[LỖI] Không tìm thấy Manifest: {MANIFEST_FILE}")
        sys.exit(1)

    df_manifest = pd.read_csv(MANIFEST_FILE)
    df_tracking = pd.read_csv(TRACKING_FILE) if TRACKING_FILE.exists() else df_manifest.copy()

    ann_dir = Path(ann_dir)
    print(f"Thư mục kiểm tra annotation: {ann_dir}")
    print(f"Tổng số ảnh Pilot: {len(df_manifest)}")

    results = []
    total_polygons = 0
    class_counts = {c: 0 for c in VALID_CLASSES}

    for idx, r in df_manifest.iterrows():
        stem = Path(r["filename"]).stem
        json_p = ann_dir / f"{stem}.json"
        img_p = Path(r["image_path"])
        if not img_p.exists():
            img_p = IMAGES_DIR / r["filename"]

        track_row = df_tracking[df_tracking["pilot_id"] == r["pilot_id"]]
        status = track_row["status"].values[0] if len(track_row) > 0 else "NOT_STARTED"

        if not json_p.exists():
            # Kiểm tra lỗi: Ảnh đã duyệt nhưng không có annotation
            if status == "APPROVED":
                issues = [{"type": "APPROVED_WITHOUT_ANNOTATION", "msg": "Ảnh được đánh dấu APPROVED nhưng không có file JSON"}]
            else:
                issues = []
            results.append({
                "pilot_id": r["pilot_id"],
                "filename": r["filename"],
                "split": r["split"],
                "status": status,
                "json_exists": False,
                "num_polygons": 0,
                "issues": issues,
            })
            continue

        issues, shapes_info = validate_single_json(json_p, img_p)

        # Kiểm tra nếu APPROVED nhưng rỗng
        if status == "APPROVED" and len(shapes_info) == 0:
            issues.append({"type": "APPROVED_EMPTY", "msg": "Ảnh APPROVED nhưng số lượng polygon = 0"})

        total_polygons += len(shapes_info)
        for s in shapes_info:
            if s["label"] in class_counts:
                class_counts[s["label"]] += 1

        results.append({
            "pilot_id": r["pilot_id"],
            "filename": r["filename"],
            "split": r["split"],
            "status": status,
            "json_exists": True,
            "num_polygons": len(shapes_info),
            "issues": issues,
        })

    # Tổng hợp báo cáo
    total_json = sum(1 for r in results if r["json_exists"])
    total_issues = sum(len(r["issues"]) for r in results)
    clean_json = sum(1 for r in results if r["json_exists"] and len(r["issues"]) == 0)

    print(f"\nTổng số file JSON hiện diện: {total_json} / {len(df_manifest)}")
    print(f"Tổng số polygon đã kiểm tra: {total_polygons}")
    print(f"Số file JSON đạt chuẩn (Clean): {clean_json} / {total_json if total_json > 0 else 1}")
    print(f"Tổng số lỗi phát hiện: {total_issues}")

    # Xuất Markdown Report
    report_lines = [
        "# BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG GÁN NHÃN (STAGE 2 QUALITY CONTROL)",
        "**Dự án:** AutoClaim-AI (`D:\\AI_CAR`)  ",
        f"**Thư mục kiểm toán:** `{ann_dir}`  ",
        f"**Thời điểm thực thi:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        "",
        "## 1. TỔNG QUAN KẾT QUẢ KIỂM TOÁN CHẤT LƯỢNG",
        "",
        "| Chỉ số kiểm soát | Giá trị đo được | Đánh giá chất lượng |",
        "| :--- | :---: | :--- |",
        f"| **Tổng số ảnh Pilot** | **{len(df_manifest)}** | 60 TRAIN / 20 VAL |",
        f"| **Số file JSON đã tạo** | **{total_json}** | {total_json/len(df_manifest)*100:.1f}% tiến độ |",
        f"| **Tổng số polygon ngoại thất** | **{total_polygons}** | Trung bình {total_polygons/total_json if total_json>0 else 0:.1f} polygon/ảnh |",
        f"| **Số file JSON hợp lệ tuyệt đối** | **{clean_json}** | Đạt {clean_json/total_json*100 if total_json>0 else 0:.1f}% |",
        f"| **Tổng số lỗi hình học / nhãn** | **{total_issues}** | {'Hoàn hảo' if total_issues==0 else 'Cần chỉnh sửa'} |",
        "",
        "---",
        "",
        "## 2. PHÂN BỐ SỐ LƯỢNG POLYGON THEO 8 LỚP",
        "",
        "| Class ID | Tên bộ phận ngoại thất | Số lượng polygon | Tỷ lệ % |",
        "| :---: | :--- | :---: | :---: |",
    ]

    for cid, cname in enumerate(VALID_CLASSES):
        cnt = class_counts[cname]
        pct = (cnt / total_polygons * 100) if total_polygons > 0 else 0.0
        report_lines.append(f"| {cid} | `{cname}` | {cnt} | {pct:.1f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. DANH SÁCH CHI TIẾT CÁC VẤN ĐỀ CẦN XỬ LÝ (ACTION ITEMS)",
        ""
    ])

    issues_found = False
    for r in results:
        if r["issues"]:
            issues_found = True
            report_lines.append(f"### Ảnh #{r['pilot_id']}: `{r['filename']}` ({r['split'].upper()}) — Trạng thái: `{r['status']}`")
            for iss in r["issues"]:
                report_lines.append(f"- **[{iss['type']}]**: {iss['msg']}")
            report_lines.append("")

    if not issues_found:
        report_lines.append("> [!TIP]\n> **XUẤT SẮC:** Không phát hiện bất kỳ lỗi cú pháp, vi phạm biên hay chồng lấn bất thường nào.")

    report_out_path = Path(report_out_path)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[OK] Đã xuất báo cáo chất lượng: {report_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm soát chất lượng Annotation")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_RAW_DIR), help="Thư mục chứa JSON cần kiểm tra")
    parser.add_argument("--report-out", type=str, default=str(DEFAULT_REPORT_OUT), help="Đường dẫn file báo cáo markdown xuất ra")
    args = parser.parse_args()

    run_qc(args.dir, args.report_out)


if __name__ == "__main__":
    main()
