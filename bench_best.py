"""
Benchmark model best.pt on test images in data/dataset/test.

Usage:
  python bench_best.py
  python bench_best.py --model best.pt --test-dir data/dataset/test --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    # Fallback for local source layout: project_root/ultralytics/ultralytics
    PROJECT_ROOT = Path(__file__).resolve().parent
    LOCAL_ULTRA_REPO = PROJECT_ROOT / "ultralytics"
    if LOCAL_ULTRA_REPO.exists():
        sys.path.insert(0, str(LOCAL_ULTRA_REPO))
    # Remove stale namespace package cache before re-import.
    sys.modules.pop("ultralytics", None)
    importlib.invalidate_caches()
    from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO best.pt on a test folder")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--test-dir", type=str, default="data/dataset/test", help="Path to test images folder")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml", help="Path to dataset YAML (for precision/recall/mAP)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split for evaluation: train/val/test")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="Device: 0, 1, cpu, ...")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup images before measuring")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of test images (0 = all)")
    return parser.parse_args()


def collect_images(test_dir: Path) -> list[Path]:
    images = [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_detection_metrics(val_results) -> None:
    """Print key detection metrics with compatibility fallbacks across Ultralytics versions."""
    precision = recall = map50 = map5095 = None

    box_metrics = getattr(val_results, "box", None)
    if box_metrics is not None:
        precision = _float_or_none(getattr(box_metrics, "mp", None))
        recall = _float_or_none(getattr(box_metrics, "mr", None))
        map50 = _float_or_none(getattr(box_metrics, "map50", None))
        map5095 = _float_or_none(getattr(box_metrics, "map", None))

    # Fallback: results_dict keys differ slightly by Ultralytics release.
    results_dict = getattr(val_results, "results_dict", None) or {}
    if precision is None:
        precision = _float_or_none(results_dict.get("metrics/precision(B)", results_dict.get("metrics/precision")))
    if recall is None:
        recall = _float_or_none(results_dict.get("metrics/recall(B)", results_dict.get("metrics/recall")))
    if map50 is None:
        map50 = _float_or_none(results_dict.get("metrics/mAP50(B)", results_dict.get("metrics/mAP50")))
    if map5095 is None:
        map5095 = _float_or_none(results_dict.get("metrics/mAP50-95(B)", results_dict.get("metrics/mAP50-95")))

    print("\nDETECTION METRICS")
    print("-" * 60)
    print(f"Precision       : {precision:.4f}" if precision is not None else "Precision       : N/A")
    print(f"Recall          : {recall:.4f}" if recall is not None else "Recall          : N/A")
    print(f"mAP@0.50        : {map50:.4f}" if map50 is not None else "mAP@0.50        : N/A")
    print(f"mAP@0.50:0.95   : {map5095:.4f}" if map5095 is not None else "mAP@0.50:0.95   : N/A")


def resolve_dataset_yaml(data_yaml: Path) -> Path:
    """Create a local dataset YAML when the original path points to cloud/non-existent root."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_path = str(cfg.get("path", "")).strip()
    current_root = Path(raw_path) if raw_path else data_yaml.parent
    looks_like_cloud_path = raw_path.startswith("/content") or raw_path.startswith("/kaggle")
    missing_local_root = not current_root.exists()

    if looks_like_cloud_path or missing_local_root:
        local_root = data_yaml.parent.resolve()
        cfg["path"] = str(local_root).replace("\\", "/")
        local_yaml = data_yaml.with_name(f"{data_yaml.stem}.local.yaml")
        with open(local_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[INFO] Dataset path in YAML is not local: '{raw_path}'")
        print(f"[INFO] Auto-generated local YAML: {local_yaml}")
        print(f"[INFO] Using dataset root: {cfg['path']}")
        return local_yaml

    return data_yaml


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    data_yaml = Path(args.data)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    if not test_dir.exists() or not test_dir.is_dir():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return 1

    images = collect_images(test_dir)
    if not images:
        print(f"[ERROR] No images found in: {test_dir}")
        return 1

    if args.max_images > 0:
        images = images[: args.max_images]

    print("=" * 60)
    print("YOLO BENCHMARK")
    print("=" * 60)
    print(f"Model     : {model_path}")
    print(f"Test dir  : {test_dir}")
    print(f"Num images: {len(images)}")
    print(f"Image size: {args.imgsz}")
    print(f"Device    : {args.device}")
    print(f"Data YAML : {data_yaml}")
    print(f"Eval split: {args.split}")
    print("=" * 60)

    model = YOLO(str(model_path))

    warmup_count = min(args.warmup, len(images))
    for i in range(warmup_count):
        _ = model.predict(
            source=str(images[i]),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

    speeds_pre = []
    speeds_inf = []
    speeds_post = []
    wall_times = []

    for img in images:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        t1 = time.perf_counter()

        wall_times.append((t1 - t0) * 1000.0)

        if results and hasattr(results[0], "speed"):
            spd = results[0].speed
            speeds_pre.append(float(spd.get("preprocess", 0.0)))
            speeds_inf.append(float(spd.get("inference", 0.0)))
            speeds_post.append(float(spd.get("postprocess", 0.0)))

    def avg(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    avg_pre = avg(speeds_pre)
    avg_inf = avg(speeds_inf)
    avg_post = avg(speeds_post)
    avg_wall = avg(wall_times)

    fps_inference = 1000.0 / avg_inf if avg_inf > 0 else 0.0
    fps_end_to_end = 1000.0 / avg_wall if avg_wall > 0 else 0.0

    print("\nRESULT")
    print("-" * 60)
    print(f"Avg preprocess : {avg_pre:.2f} ms")
    print(f"Avg inference  : {avg_inf:.2f} ms")
    print(f"Avg postprocess: {avg_post:.2f} ms")
    print(f"Avg end-to-end : {avg_wall:.2f} ms")
    print(f"FPS (inference): {fps_inference:.2f}")
    print(f"FPS (end-to-end): {fps_end_to_end:.2f}")

    if data_yaml.exists():
        resolved_data_yaml = resolve_dataset_yaml(data_yaml)
        print("\nRunning validation for precision/recall/mAP...")
        val_results = model.val(
            data=str(resolved_data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        print_detection_metrics(val_results)
    else:
        print("\n[WARN] Data YAML not found, skip precision/recall/mAP evaluation.")
        print(f"       Expected at: {data_yaml}")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
