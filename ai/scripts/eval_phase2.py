import sys
import json
import time
from pathlib import Path
import cv2

# Set UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from ai.inference.pipeline import TwoStagePipeline

def main():
    car_dir = repo_root / "car"
    out_dir = repo_root / "ai" / "reports" / "eval_real_cars_phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    damage_model = repo_root / "ai" / "models" / "weights" / "damage_best.pt"
    parts_model = repo_root / "ai" / "models" / "weights" / "parts_seg.pt"

    print("=== KHỞI TẠO TWO-STAGE PIPELINE (PHASE 2 OPTIMIZED) ===")
    pipeline = TwoStagePipeline(
        damage_model_path=damage_model,
        parts_model_path=parts_model,
        damage_conf=0.25,
        parts_conf=0.30,
        iou=0.45,
        imgsz=640,
    )

    image_paths = sorted(car_dir.glob("*.png")) + sorted(car_dir.glob("*.jpg"))
    # Eliminate duplicate path objects if any
    image_paths = sorted(list({p.resolve(): p for p in image_paths}.values()), key=lambda p: p.name)

    print(f"Tìm thấy {len(image_paths)} ảnh trong {car_dir}\n")

    results = []

    for img_path in image_paths:
        print(f"--- Đang xử lý: {img_path.name} ---")
        t0 = time.time()
        res = pipeline.run(str(img_path))
        elapsed = round(time.time() - t0, 3)

        out_annotated_path = out_dir / f"annotated_{img_path.name}"
        pipeline.visualize(str(img_path), res, output_path=out_annotated_path)

        entry = {
            "file_name": img_path.name,
            "image_hw": [res.image_height, res.image_width],
            "status": res.status,
            "shot_type": res.shot_type,
            "action_required": res.action_required,
            "num_damages": len(res.detections),
            "num_parts": len(res.parts_detected),
            "elapsed_sec": elapsed,
            "detections": [
                {
                    "damage_type": d["damage_type"],
                    "confidence": round(d["confidence"], 4),
                    "part": d["part"],
                    "located_on_part": d["located_on_part"],
                    "match_method": d["match_method"],
                    "warning": d.get("warning", ""),
                    "action_required": d.get("action_required", ""),
                    "bbox": [round(v, 1) for v in d["bbox_xyxy"]],
                }
                for d in res.detections
            ],
            "parts": [
                {
                    "part_name": p.get("part_name", p.get("part_key")),
                    "confidence": round(p.get("confidence", 0.0), 4),
                    "bbox": [round(v, 1) for v in p.get("bbox_xyxy", [])] if p.get("bbox_xyxy") else [],
                }
                for p in res.parts_detected
            ]
        }
        results.append(entry)

        print(f"  > Status: {res.status}")
        print(f"  > Action Required: {res.action_required}")
        print(f"  > Damages ({len(res.detections)}):")
        for idx, d in enumerate(res.detections):
            warn = f" | Warning: {d['warning']}" if d.get('warning') else ""
            print(f"     [{idx+1}] {d['damage_type']} -> {d['part']} (conf: {d['confidence']:.2f}, method: {d['match_method']}){warn}")
        print(f"  > Parts ({len(res.parts_detected)}):")
        for idx, p in enumerate(res.parts_detected):
            print(f"     [{idx+1}] {p.get('part_name')} (conf: {p.get('confidence', 0.0):.2f})")
        print()

    eval_json_path = out_dir / "real_cars_evaluation_phase2.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n Đã lưu kết quả đánh giá Phase 2 tại: {eval_json_path}")
    print(f" Đã lưu {len(results)} ảnh annotated tại: {out_dir}")

if __name__ == "__main__":
    main()
