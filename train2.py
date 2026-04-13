import os
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = r"D:\face\datasets\human\data.yaml"
MODEL_PRETRAIN = "yolov8s-seg.pt"
PROJECT_DIR = r"runs\segment"
RUN_NAME = "my-seg"

EPOCHS = 40
IMG_SIZE = 640

BATCH = 8          # ringan RAM/VRAM, imgsz tetap 640
WORKERS = 0        # Windows: paling stabil
DEVICE = 0         # 0 untuk GPU, atau "cpu"
CACHE = "disk"     # kalau error, ganti ke False

RESUME = False

def check_data_yaml(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {p.resolve()}")
    print(f"[OK] data.yaml: {p.resolve()}")

def main():
    check_data_yaml(DATA_YAML)

    if RESUME:
        raise ValueError("RESUME=True tapi path last.pt belum kamu set.")
    else:
        model = YOLO(MODEL_PRETRAIN)

    results = model.train(
        task="segment",
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,      # tetap 640
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        cache=CACHE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        amp=True,            # aman kalau GPU, biasanya bantu (VRAM)
    )

    run_dir = Path(PROJECT_DIR) / RUN_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("\n==== TRAINING DONE ====")
    print(f"Run folder : {run_dir.resolve()}")
    print(f"Best weight: {best_pt.resolve() if best_pt.exists() else best_pt}")
    print(f"Last weight: {last_pt.resolve() if last_pt.exists() else last_pt}")

if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    main()
