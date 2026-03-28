"""
Run YOLO validation on local test split with correct Windows/local paths.
Equivalent to:
  yolo val model=best.pt data=data/dataset/data.local.yaml split=test imgsz=512 plots=True
"""

from __future__ import annotations

import argparse
import importlib
import sys
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
    sys.modules.pop("ultralytics", None)
    importlib.invalidate_caches()
    from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate best.pt on test split")
    parser.add_argument("--model", default="best.pt", help="Model path")
    parser.add_argument("--data", default="data/dataset/data.local.yaml", help="Dataset YAML path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--device", default="0", help="Device: 0, 1, cpu, ...")
    parser.add_argument("--plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Disable plots")
    return parser.parse_args()


def resolve_dataset_yaml(data_yaml: Path) -> Path:
    """Create data.local.yaml if YAML points to cloud path or missing root."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_path = str(cfg.get("path", "")).strip()
    root = Path(raw_path) if raw_path else data_yaml.parent
    looks_like_cloud = raw_path.startswith("/content") or raw_path.startswith("/kaggle")

    if looks_like_cloud or not root.exists():
        cfg["path"] = str(data_yaml.parent.resolve()).replace("\\", "/")
        local_yaml = data_yaml.with_name(f"{data_yaml.stem}.local.yaml")
        with open(local_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        return local_yaml

    return data_yaml


def metric_value(val_results, key: str):
    d = getattr(val_results, "results_dict", None) or {}
    return d.get(key)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    # If data.local.yaml is missing, fallback to data.yaml then auto-localize.
    if not data_path.exists() and data_path.name == "data.local.yaml":
        fallback = data_path.with_name("data.yaml")
        if fallback.exists():
            data_path = fallback

    data_path = resolve_dataset_yaml(data_path)

    print("=" * 60)
    print("YOLO VAL (LOCAL)")
    print("=" * 60)
    print(f"Model : {model_path}")
    print(f"Data  : {data_path}")
    print(f"Split : {args.split}")
    print(f"imgsz : {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Plots : {args.plots}")
    print("=" * 60)

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        plots=args.plots,
        device=args.device,
    )

    p = metric_value(results, "metrics/precision(B)")
    r = metric_value(results, "metrics/recall(B)")
    m50 = metric_value(results, "metrics/mAP50(B)")
    m5095 = metric_value(results, "metrics/mAP50-95(B)")

    print("\nMETRICS")
    print("-" * 60)
    print(f"Precision     : {p:.4f}" if isinstance(p, (int, float)) else f"Precision     : {p}")
    print(f"Recall        : {r:.4f}" if isinstance(r, (int, float)) else f"Recall        : {r}")
    print(f"mAP@0.50      : {m50:.4f}" if isinstance(m50, (int, float)) else f"mAP@0.50      : {m50}")
    print(f"mAP@0.50:0.95 : {m5095:.4f}" if isinstance(m5095, (int, float)) else f"mAP@0.50:0.95 : {m5095}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
