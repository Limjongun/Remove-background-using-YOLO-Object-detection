"""
REMOVE BACKGROUND (HUMAN) pakai YOLOv8-Seg (batch folder gambar)
+ OPSI OUTPUT 1080p (1920x1080) setelah background dihapus

Output: PNG transparan (RGBA) di folder output.

Install:
  pip install ultralytics opencv-python

Atur:
  SEG_WEIGHTS  -> path model seg kamu
  INPUT_DIR    -> folder berisi banyak gambar
  OUTPUT_DIR   -> folder hasil

Run:
  python remove_bg_human_folder_1080p.py
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

# ================== KONFIGURASI ==================
SEG_WEIGHTS = r"D:\face\runs\segment\my-seg6\weights\best.pt"

INPUT_DIR  = r"D:\face\images"                 # folder input gambar
OUTPUT_DIR = r"D:\face\output_no_bg_png"      # folder output PNG

CONF = 0.35
IMG_SIZE = 640
DEVICE = 0  # 0 untuk GPU, atau "cpu"

# Target class untuk manusia (sesuaikan jika label dataset kamu beda)
TARGET_CLASS_NAMES = {"person", "human", "people"}

# Postprocess biar tepi lebih rapi
MORPH_KERNEL = 3   # 0 untuk matikan; 3/5/7 umum
FEATHER_BLUR = 5   # 0 untuk matikan; 3/5 bagus untuk smoothing

# ===== OUTPUT 1080p =====
FORCE_1080P = True
TARGET_W, TARGET_H = 1920, 1080                # 1080p
KEEP_ASPECT_WITH_PADDING = True                # True = letterbox ke 1920x1080, False = stretch

# Ekstensi gambar yang diproses
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_images(folder: str):
    folder = Path(folder)
    files = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            files.append(f)
    return sorted(files)


def resize_rgba_to_1080p(rgba: np.ndarray, target_w=1920, target_h=1080, keep_aspect=True) -> np.ndarray:
    """
    Upscale/resize hasil RGBA ke 1080p.
    - keep_aspect=True  : resize proporsional + padding transparan (letterbox)
    - keep_aspect=False : stretch langsung ke 1920x1080
    """
    h, w = rgba.shape[:2]

    if not keep_aspect:
        return cv2.resize(rgba, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # canvas transparan
    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_union_mask(result, orig_hw):
    """
    result: Ultralytics Results (1 image)
    orig_hw: (H, W) gambar asli
    return: mask union float32 0..1 ukuran (H, W)
    """
    H, W = orig_hw
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    masks = result.masks.data
    if masks is None or len(masks) == 0:
        return None

    masks_np = masks.detach().cpu().numpy().astype(np.float32)     # (N, mh, mw)
    cls_np = result.boxes.cls.detach().cpu().numpy().astype(int)   # (N,)
    names = result.names  # dict id->name

    # ambil mask yang class-nya manusia
    human_idxs = []
    for i, cid in enumerate(cls_np):
        cname = str(names.get(int(cid), "")).lower()
        if cname in TARGET_CLASS_NAMES:
            human_idxs.append(i)

    # fallback: kalau model cuma 1 class / nama class beda, pakai semua mask
    use_idxs = human_idxs if len(human_idxs) > 0 else list(range(masks_np.shape[0]))

    union = None
    for i in use_idxs:
        m = masks_np[i]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        union = m if union is None else np.maximum(union, m)

    return np.clip(union, 0.0, 1.0)


def postprocess_mask(mask01: np.ndarray):
    """
    mask01: float 0..1
    return: float 0..1 (lebih halus/rapi)
    """
    m = (mask01 * 255).astype(np.uint8)

    if MORPH_KERNEL and MORPH_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    if FEATHER_BLUR and FEATHER_BLUR > 0:
        b = FEATHER_BLUR if FEATHER_BLUR % 2 == 1 else FEATHER_BLUR + 1
        m = cv2.GaussianBlur(m, (b, b), 0)

    return (m.astype(np.float32) / 255.0)


def remove_bg_and_save(model: YOLO, img_path: Path, out_dir: str):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[SKIP] Gagal baca: {img_path}")
        return

    H, W = img_bgr.shape[:2]

    r = model.predict(img_bgr, conf=CONF, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]

    mask01 = build_union_mask(r, (H, W))

    # Siapkan RGBA dasar dari gambar asli
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    if mask01 is None:
        # tidak ada deteksi → alpha full
        rgba[:, :, 3] = 255
        suffix = "_nobg_1080p.png" if FORCE_1080P else "_nobg.png"
        if FORCE_1080P:
            rgba = resize_rgba_to_1080p(rgba, TARGET_W, TARGET_H, KEEP_ASPECT_WITH_PADDING)
        out_path = Path(out_dir) / (img_path.stem + suffix)
        cv2.imwrite(str(out_path), rgba)
        print(f"[NO DET] {img_path.name} -> {out_path.name}")
        return

    mask01 = postprocess_mask(mask01)
    alpha = (mask01 * 255).astype(np.uint8)
    rgba[:, :, 3] = alpha

    # ===== Resize ke 1080p setelah background dihapus =====
    if FORCE_1080P:
        rgba = resize_rgba_to_1080p(rgba, TARGET_W, TARGET_H, KEEP_ASPECT_WITH_PADDING)
        out_path = Path(out_dir) / (img_path.stem + "_nobg_1080p.png")
    else:
        out_path = Path(out_dir) / (img_path.stem + "_nobg.png")

    cv2.imwrite(str(out_path), rgba)
    print(f"[OK] {img_path.name} -> {out_path.name}")


def main():
    if not os.path.exists(SEG_WEIGHTS):
        raise FileNotFoundError(f"weights tidak ditemukan: {SEG_WEIGHTS}")

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"folder input tidak ditemukan: {INPUT_DIR}")

    ensure_dir(OUTPUT_DIR)

    model = YOLO(SEG_WEIGHTS)

    images = list_images(INPUT_DIR)
    if not images:
        print(f"Tidak ada gambar di: {INPUT_DIR}")
        return

    print(f"Total gambar: {len(images)}")
    print("Mulai remove background...")

    for i, p in enumerate(images, 1):
        remove_bg_and_save(model, p, OUTPUT_DIR)
        if i % 20 == 0:
            print(f"Progress: {i}/{len(images)}")

    print("Selesai.")
    print(f"Hasil ada di: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
