"""
SETUP SCRIPT CHO KAGGLE NOTEBOOKS
Flow mặc định:
1) Clone project vào /kaggle/working/YOLOv12-Attention
2) Dataset tải bằng gdown và chuẩn hóa về /kaggle/working/datasets/recycle/data.yaml
3) Train với yolo12m-attn.yaml có CBAM + SimAM
"""

import os
import subprocess
import zipfile
import shutil
from pathlib import Path


def run(cmd: str, cwd: str = None):
    """Run shell command and fail fast if command exits non-zero."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def prepare_dataset_from_gdown(file_id: str, working_dir: Path) -> Path:
    """Download + unzip + normalize Roboflow dataset structure for Kaggle."""
    datasets_dir = working_dir / "datasets"
    recycle_dir = datasets_dir / "recycle"
    zip_path = working_dir / "dataset.zip"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    recycle_dir.mkdir(parents=True, exist_ok=True)

    run(f"gdown {file_id} -O {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(datasets_dir)

    # Handle common nested layout: /datasets/recycle/recycle/*
    nested_recycle = recycle_dir / "recycle"
    if nested_recycle.exists():
        for item in nested_recycle.iterdir():
            target = recycle_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(nested_recycle, ignore_errors=True)

    data_yaml = recycle_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml sau khi giải nén: {data_yaml}")

    return data_yaml


# ============= Step 1: Base packages =============
run("pip install -q opencv-python pyyaml tqdm matplotlib scipy einops timm gdown")

# ============= Step 2: Paths =============

WORKING = Path("/kaggle/working")
PROJECT_ROOT = WORKING / "YOLOv12-Attention"
ULTRA_ROOT = PROJECT_ROOT / "ultralytics"
MODEL_YAML = ULTRA_ROOT / "ultralytics" / "yolo12m-attn.yaml"
RUNS_DIR = WORKING / "yolo_runs"
DATASET_FILE_ID = "1GPO3zGVzbVfXH5euqbV2MDtzMABE2EgK"

print(f"Project root: {PROJECT_ROOT}")
print(f"Model yaml : {MODEL_YAML}")

if not PROJECT_ROOT.exists():
    raise FileNotFoundError("Không tìm thấy project. Hãy clone vào /kaggle/working/YOLOv12-Attention")
if not MODEL_YAML.exists():
    raise FileNotFoundError("Không tìm thấy file model yaml yolo12m-attn.yaml")

# ============= Step 2.1: Prepare dataset =============
DATA_YAML = prepare_dataset_from_gdown(DATASET_FILE_ID, WORKING)
print(f"Data yaml  : {DATA_YAML}")

# ============= Step 3: Force dùng source local ultralytics (để nhận CBAM/SimAM) =============
run("pip uninstall -y ultralytics")
run("pip install -e .", cwd=str(ULTRA_ROOT))

# ============= Step 4: Verify attention modules =============
run("python -c \"from ultralytics.nn.modules import CBAM, SimAM; print('OK: CBAM/SimAM loaded')\"")

# ============= Step 5: Train (theo cấu hình bạn gửi) =============
run(
    "yolo train "
    "model=/kaggle/working/YOLOv12-Attention/ultralytics/ultralytics/yolo12m-attn.yaml "
    "data=/kaggle/working/datasets/recycle/data.yaml "
    "epochs=50 imgsz=512 batch=16 workers=4 cache=disk "
    "device=0 amp=True freeze=0 pretrained=False "
    "project=/kaggle/working/yolo_runs name=attn_50_fix"
)
