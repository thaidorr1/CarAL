"""Train CarDD damage detection/segmentation; ho tro ca Detection (yolov8n.pt) va
Segmentation (yolov8n-seg.pt). Model task duoc tu dong nhan dien tu file weights.
Chon best.pt theo validation loss, khong dung test set de chon model.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime import AI_ROOT, prepare_dataset, select_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--model', type=Path, default=AI_ROOT / 'models' / 'yolov8n.pt',
        help='Weights file: yolov8n.pt cho Detection, yolov8n-seg.pt cho Segmentation',
    )
    p.add_argument('--data', type=Path, default=AI_ROOT / 'configs/data.yaml')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='auto')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument(
        '--name', default='cardd_detect_30epochs',
        help='Ten thu muc luu ket qua trong runs/damage/',
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exist-ok', action='store_true', default=True, help='Ghi de hoac tiep tuc run neu da ton tai')
    a = p.parse_args()

    # --- Validate numeric args ---
    if a.epochs <= 0 or a.imgsz <= 0 or a.batch <= 0:
        p.error('epochs, imgsz and batch must be positive')
    if Path(a.name).name != a.name or a.name in {'.', '..'}:
        p.error('name must be a single directory name')
    if not a.model.is_file():
        p.error(f'Checkpoint does not exist: {a.model}')

    # --- Load model; task duoc tu dong nhan dien (detect / segment) ---
    from ultralytics import YOLO
    model = YOLO(str(a.model.resolve()))
    task = model.task  # 'detect' hoac 'segment'
    print(f'[train_damage] Model task : {task}')
    print(f'[train_damage] Weights    : {a.model.name}')

    # Canh bao neu dung seg-model tren dataset bbox-only
    if task == 'segment':
        print(
            '[train_damage] CANH BAO: task=segment yeu cau nhan co polygon mask.\n'
            '  Neu labels/ chi chua bbox (5 cot) hay doi sang --model yolov8n.pt (detect).'
        )

    run = AI_ROOT / 'runs/damage' / a.name

    dataset = prepare_dataset(a.data, run / 'dataset.yaml')

    # exist_ok chi cho folder moi vua tao o tren; khong ghi de thi nghiem cu.
    model.train(
        data=str(dataset),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=select_device(a.device),
        workers=a.workers,
        cache=False,
        patience=a.patience,
        project=str(run.parent),
        name=a.name,
        exist_ok=True,
        seed=a.seed,
        plots=True,
    )

    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training ended without best.pt')
    print(f'[train_damage] Validation-selected checkpoint: {best}')


if __name__ == '__main__':
    main()
