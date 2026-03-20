"""
Quick setup cho Colab (phiên bản gọn)
"""

import subprocess
import importlib
import zipfile
import shutil
from pathlib import Path


def run(cmd: str, cwd: str = None):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def prepare_dataset(file_id: str) -> Path:
    root = Path("/content")
    datasets = root / "datasets"
    recycle = datasets / "recycle"
    zip_path = root / "dataset.zip"

    datasets.mkdir(parents=True, exist_ok=True)
    recycle.mkdir(parents=True, exist_ok=True)

    run(f"gdown {file_id} -O {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(datasets)

    nested = recycle / "recycle"
    if nested.exists():
        for item in nested.iterdir():
            target = recycle / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(nested, ignore_errors=True)

    data_yaml = recycle / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_yaml}")
    return data_yaml


try:
    drive = importlib.import_module("google.colab.drive")
except ModuleNotFoundError as e:
    raise RuntimeError("Script này cần chạy trên Google Colab") from e

drive.mount("/content/drive")

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Du_An_Ca_Nhan/Computer_Vision/phan_loai_vat_lieu_tai_che/Attention"
)
ULTRA_ROOT = PROJECT_ROOT / "ultralytics"
MODEL_YAML = ULTRA_ROOT / "ultralytics" / "yolo12m-attn.yaml"
DATA_YAML = prepare_dataset("1GPO3zGVzbVfXH5euqbV2MDtzMABE2EgK")
RUNS_DIR = PROJECT_ROOT / "runs" / "colab"

run("pip install -q torch torchvision")
run("pip install -q opencv-python pyyaml tqdm matplotlib scipy einops timm gdown")
run("pip uninstall -y ultralytics")
run("pip install -e .", cwd=str(ULTRA_ROOT))
run("python -c \"from ultralytics.nn.modules import CBAM, SimAM; print('OK: CBAM/SimAM loaded')\"")

run(
    "yolo train "
    f"model={MODEL_YAML} "
    f"data={DATA_YAML} "
    "epochs=50 imgsz=512 batch=16 workers=4 cache=disk "
    "device=0 amp=True freeze=0 pretrained=False "
    f"project={RUNS_DIR} name=attn_50_fix"
)

print("✓ Done!")
