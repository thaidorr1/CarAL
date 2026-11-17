import json
from pathlib import Path
import shutil
import sys

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import AI_ROOT, canonical_name

RAW = AI_ROOT / "data/raw/CarDD"
OUT = AI_ROOT / "data/yolo"
SPLITS = ("train", "val", "test")


def norm(name: str) -> str:
    # Tên model dùng '_' để tương thích checkpoint cũ; inference nhận cả hai dạng.
    return canonical_name(name).replace(" ", "_")


def load(split: str) -> dict:
    p = RAW / "annotations" / f"instances_{split}.json"
    if not p.is_file():
        p = RAW / "annotations" / f"instances_{split}2017.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    cats = sorted(load("train")["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [norm(c["name"]) for c in cats]

    for split in SPLITS:
        data = load(split)
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # tìm ảnh theo tên file, phòng trường hợp có thư mục con
        index = {p.name: p for p in (RAW / split).rglob("*") if p.is_file()}
        lines: dict[int, list[str]] = {i["id"]: [] for i in data["images"]}
        images = {i["id"]: i for i in data["images"]}
        skipped = 0

        for a in data["annotations"]:
            seg = a.get("segmentation")
            if not isinstance(seg, list):  # RLE (iscrowd) -> bỏ qua
                skipped += 1
                continue
            img = images[a["image_id"]]
            w, h = img["width"], img["height"]
            # NOTE REVIEW: Đang xuất YOLO segmentation (class + các điểm polygon),
            # không phải bbox detection (class + cx cy width height).
            # Cần chọn model/train task tương ứng khi triển khai huấn luyện.
            for poly in seg:
                if len(poly) < 6:  # cần tối thiểu 3 điểm
                    skipped += 1
                    continue
                pts = [
                    f"{min(max(v / (w if k % 2 == 0 else h), 0), 1):.6f}"
                    for k, v in enumerate(poly)
                ]
                lines[a["image_id"]].append(f"{cat_to_idx[a['category_id']]} " + " ".join(pts))

        missing = 0
        for img_id, img in images.items():
            file_name = Path(img["file_name"]).name
            file_stem = Path(img["file_name"]).stem
            src = index.get(file_name)
            if src is not None:
                dst = img_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
            else:
                missing += 1
            # Luôn xuất nhãn polygon YOLO để sẵn sàng bộ nhãn chuẩn
            (lbl_dir / f"{file_stem}.txt").write_text("\n".join(lines[img_id]), encoding="utf-8")

        print(f"{split}: {len(images)} ảnh, thiếu file {missing}, bỏ qua {skipped} polygon")

    resolved_path = str(OUT.resolve()).replace("\\", "/")
    yaml = [f"path: {resolved_path}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (AI_ROOT / "configs").mkdir(exist_ok=True)
    (AI_ROOT / "configs/data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print("classes:", dict(enumerate(names)))


if __name__ == "__main__":
    main()
