"""
SETUP SCRIPT CHO GOOGLE COLAB
Flow mặc định:
1) Code nằm ở Google Drive: .../Attention
2) Dataset tải bằng gdown về /content/datasets/recycle/data.yaml
3) Force dùng source local ultralytics để nhận CBAM + SimAM
"""

import subprocess
import importlib
import zipfile
import shutil
from pathlib import Path


def run(cmd: str, cwd: str = None):
    """Run shell command and fail fast if command exits non-zero."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def prepare_dataset_from_gdown(file_id: str) -> Path:
    """Download + unzip + normalize Roboflow dataset structure for Colab."""
    working = Path("/content")
    datasets_dir = working / "datasets"
    recycle_dir = datasets_dir / "recycle"
    zip_path = working / "dataset.zip"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    recycle_dir.mkdir(parents=True, exist_ok=True)

    run(f"gdown {file_id} -O {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(datasets_dir)

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


# ============= Step 1: Mount Google Drive =============
try:
    drive = importlib.import_module("google.colab.drive")
except ModuleNotFoundError as e:
    raise RuntimeError("Script này cần chạy trên Google Colab") from e

drive.mount("/content/drive")


# ============= Step 2: Paths =============
PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Du_An_Ca_Nhan/Computer_Vision/phan_loai_vat_lieu_tai_che/Attention"
)
ULTRA_ROOT = PROJECT_ROOT / "ultralytics"
MODEL_YAML = ULTRA_ROOT / "ultralytics" / "yolo12m-attn.yaml"
DATA_YAML = PROJECT_ROOT / "data" / "dataset" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "colab"
DATASET_FILE_ID = "1GPO3zGVzbVfXH5euqbV2MDtzMABE2EgK"

print(f"Project root: {PROJECT_ROOT}")
print(f"Model yaml : {MODEL_YAML}")

if not PROJECT_ROOT.exists():
    raise FileNotFoundError("Không tìm thấy project trong Google Drive")
if not MODEL_YAML.exists():
    raise FileNotFoundError("Không tìm thấy yolo12m-attn.yaml")


# ============= Step 3: Install dependencies =============
run("pip install -q torch torchvision")
run("pip install -q opencv-python pyyaml tqdm matplotlib scipy einops timm gdown")


# ============= Step 3.1: Prepare dataset in /content/datasets/recycle =============
DATA_YAML = prepare_dataset_from_gdown(DATASET_FILE_ID)
print(f"Data yaml  : {DATA_YAML}")


# ============= Step 4: Force local editable ultralytics =============
run("pip uninstall -y ultralytics")
run("pip install -e .", cwd=str(ULTRA_ROOT))


# ============= Step 5: Verify CBAM/SimAM =============
run("python -c \"from ultralytics.nn.modules import CBAM, SimAM; print('OK: CBAM/SimAM loaded')\"")


# ============= Step 6: Train =============
run(
    "yolo train "
    f"model={MODEL_YAML} "
    f"data={DATA_YAML} "
    "epochs=50 imgsz=512 batch=16 workers=4 cache=disk "
    "device=0 amp=True freeze=0 pretrained=False "
    f"project={RUNS_DIR} name=attn_50_fix"
)

print("✓ Training completed!")
