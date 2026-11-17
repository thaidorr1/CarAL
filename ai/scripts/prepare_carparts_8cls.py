"""
Script chuẩn hóa và ánh xạ dataset Ultralytics carparts-seg sang 8 lớp AutoClaim-AI.
Output directory: ai/data/parts_carparts_8cls/
Taxonomy Remapping:
  0: can_truoc       <- [8] (front_bumper)
  1: can_sau         <- [0] (back_bumper)
  2: nap_capo        <- [16] (hood)
  3: ve_che_banh     <- [] (giữ trống ID 3)
  4: cua_truoc       <- [9, 11, 14] (front_door, front_left_door, front_right_door)
  5: cua_sau         <- [1, 3, 6] (back_door, back_left_door, back_right_door)
  6: kinh_xe         <- [2, 10] (back_glass, front_glass)
  7: den_xe          <- [4, 5, 7, 12, 13, 15] (back_left_light, back_light, back_right_light, front_left_light, front_light, front_right_light)
  Dropped: [17, 18, 19, 20, 21, 22] (left_mirror, object, right_mirror, tailgate, trunk, wheel)

QC:
  - Bỏ qua polygon suy biến (< 3 đỉnh, tức < 6 tọa độ)
  - Clamp tọa độ về [0.0, 1.0]
  - Giữ lại file .txt rỗng (ảnh nền) nếu không còn nhãn nào
"""

import os
import sys
import glob
import shutil
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = os.path.abspath("scratch/downloads/carparts-seg")
DST_DIR = os.path.abspath("ai/data/parts_carparts_8cls")

CLASS_MAPPING = {
    8: 0,   # front_bumper -> can_truoc
    0: 1,   # back_bumper -> can_sau
    16: 2,  # hood -> nap_capo
    # 3: ve_che_banh (trống)
    9: 4,   # front_door -> cua_truoc
    11: 4,  # front_left_door -> cua_truoc
    14: 4,  # front_right_door -> cua_truoc
    1: 5,   # back_door -> cua_sau
    3: 5,   # back_left_door -> cua_sau
    6: 5,   # back_right_door -> cua_sau
    2: 6,   # back_glass -> kinh_xe
    10: 6,  # front_glass -> kinh_xe
    4: 7,   # back_left_light -> den_xe
    5: 7,   # back_light -> den_xe
    7: 7,   # back_right_light -> den_xe
    12: 7,  # front_left_light -> den_xe
    13: 7,  # front_light -> den_xe
    15: 7,  # front_right_light -> den_xe
}

DROPPED_CLASSES = {17, 18, 19, 20, 21, 22}

CLASS_NAMES_8 = {
    0: "can_truoc",
    1: "can_sau",
    2: "nap_capo",
    3: "ve_che_banh",
    4: "cua_truoc",
    5: "cua_sau",
    6: "kinh_xe",
    7: "den_xe"
}

def remap_dataset():
    print("=" * 65)
    print("   AUTOCLAIM-AI — CARPARTS-SEG 8-CLASS REMAPPING PIPELINE")
    print("=" * 65)
    print(f"Thư mục nguồn: {SRC_DIR}")
    print(f"Thư mục đích:   {DST_DIR}")
    
    # Locate dataset root inside extract dir (in case zip has a subfolder)
    base_src = SRC_DIR
    if not os.path.exists(os.path.join(base_src, "images")):
        # Look for subfolder
        subdirs = [os.path.join(base_src, d) for d in os.listdir(base_src) if os.path.isdir(os.path.join(base_src, d))]
        for sd in subdirs:
            if os.path.exists(os.path.join(sd, "images")):
                base_src = sd
                break
                
    print(f"Thư mục gốc dữ liệu tìm thấy: {base_src}")
    
    splits = ["train", "val", "test"]
    stats = {}
    
    total_degenerate = []
    total_dropped_instances = 0
    
    for split in splits:
        src_img_dir = os.path.join(base_src, "images", split)
        src_lbl_dir = os.path.join(base_src, "labels", split)
        
        dst_img_dir = os.path.join(DST_DIR, "images", split)
        dst_lbl_dir = os.path.join(DST_DIR, "labels", split)
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not os.path.exists(src_img_dir):
            print(f"[CẢNH BÁO] Không tìm thấy thư mục {src_img_dir}")
            continue
            
        img_files = sorted(glob.glob(os.path.join(src_img_dir, "*.*")))
        print(f"\n--- Xử lý phân tập [{split.upper()}]: {len(img_files)} ảnh ---")
        
        split_stats = {
            "total_images": len(img_files),
            "background_images": 0,
            "annotated_images": 0,
            "class_counts": defaultdict(int),
            "dropped_counts": defaultdict(int),
            "degenerate_polygons": 0
        }
        
        for img_path in img_files:
            img_name = os.path.basename(img_path)
            stem, _ = os.path.splitext(img_name)
            dst_img_path = os.path.join(dst_img_dir, img_name)
            
            # Link or copy image
            if not os.path.exists(dst_img_path):
                try:
                    os.link(img_path, dst_img_path)
                except Exception:
                    shutil.copy2(img_path, dst_img_path)
                    
            # Process label file
            src_lbl_path = os.path.join(src_lbl_dir, f"{stem}.txt")
            dst_lbl_path = os.path.join(dst_lbl_dir, f"{stem}.txt")
            
            valid_lines = []
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        parts = line_str.split()
                        try:
                            orig_cls = int(parts[0])
                        except ValueError:
                            continue
                            
                        if orig_cls in DROPPED_CLASSES:
                            split_stats["dropped_counts"][orig_cls] += 1
                            total_dropped_instances += 1
                            continue
                            
                        if orig_cls not in CLASS_MAPPING:
                            split_stats["dropped_counts"][orig_cls] += 1
                            total_dropped_instances += 1
                            continue
                            
                        new_cls = CLASS_MAPPING[orig_cls]
                        coords = [float(x) for x in parts[1:]]
                        
                        # Check degenerate polygon (< 3 vertices, i.e. < 6 numbers)
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            split_stats["degenerate_polygons"] += 1
                            total_degenerate.append((f"{split}/{stem}.txt", line_idx, orig_cls, len(coords)))
                            continue
                            
                        # Clamp coords to [0.0, 1.0]
                        clamped = [min(1.0, max(0.0, c)) for c in coords]
                        split_stats["class_counts"][new_cls] += 1
                        
                        coord_str = " ".join(f"{c:.6f}" for c in clamped)
                        valid_lines.append(f"{new_cls} {coord_str}\n")
                        
            # Write valid lines or empty file
            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)
                
            if len(valid_lines) > 0:
                split_stats["annotated_images"] += 1
            else:
                split_stats["background_images"] += 1
                
        stats[split] = split_stats
        
    print("\n" + "=" * 65)
    print("            BÁO CÁO THỐNG KÊ HOÀN THÀNH ÁNH XẠ")
    print("=" * 65)
    
    for split in splits:
        if split not in stats:
            continue
        s = stats[split]
        print(f"\nPhân tập: {split.upper()}")
        print(f"  - Tổng số ảnh:           {s['total_images']}")
        print(f"  - Ảnh có nhãn hữu hiệu:  {s['annotated_images']}")
        print(f"  - Ảnh nền (nhãn rỗng):   {s['background_images']}")
        print(f"  - Đa giác suy biến loại: {s['degenerate_polygons']}")
        print("  - Phân bố instance theo 8 lớp:")
        for cid in range(8):
            cname = CLASS_NAMES_8[cid]
            cnt = s["class_counts"][cid]
            print(f"     Class {cid} ({cname:12s}): {cnt:6d} instances")
            
    print(f"\nTổng instance ngoài phạm vi đã lọc bỏ: {total_dropped_instances}")
    print(f"Tổng đa giác suy biến (<3 đỉnh) đã loại: {len(total_degenerate)}")
    return stats, total_degenerate

if __name__ == "__main__":
    remap_dataset()
