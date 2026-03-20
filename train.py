"""
Script huấn luyện YOLO12 với Attention Mechanisms
Tích hợp CBAM và SimAM để phân loại vật liệu tái chế
"""
import os
import sys
import importlib
import torch
import yaml
from pathlib import Path

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

# ============= CẤU HÌNH HUẤN LUYỆN =============
CONFIG = {
    # Model configuration
    "model": "ultralytics/ultralytics/yolo12m-attn.yaml",  # Đường dẫn local chuẩn
    "device": 0,  # GPU ID mặc định, script sẽ tự fallback về CPU nếu không có CUDA
    
    # Dataset configuration
    "data": "data/dataset/data.yaml",  # Dataset bạn đã copy vào /data/dataset
    
    # Training parameters
    "epochs": 50,
    "imgsz": 512,
    "batch": 16,  # Giảm nếu lỗi out of memory
    "patience": 20,  # Early stopping
    "freeze": 0,  # Không freeze layer để CBAM/SimAM học đầy đủ
    "pretrained": False,
    "save": True,
    "device": 0,  # GPU device
    
    # Optimization
    "optimizer": "SGD",  # SGD, Adam, AdamW
    "lr0": 0.01,  # Initial learning rate
    "lrf": 0.01,  # Final learning rate
    "momentum": 0.937,
    "weight_decay": 0.0005,
    
    # Data augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    
    # Validation
    "val": True,
    "split": 0.1,  # 10% validation
    
    # Logging
    "verbose": True,
    "plots": True,
}

def check_environment():
    """Kiểm tra môi trường PyTorch"""
    print("=" * 60)
    print("KIỂM TRA MÔI TRƯỜNG HUẤN LUYỆN")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

def test_attention_modules():
    """Kiểm tra xem các module attention đã được load đúng"""
    print("=" * 60)
    print("KIỂM TRA CÁC MODULE ATTENTION")
    print("=" * 60)
    try:
        from ultralytics.nn.modules import CBAM, SimAM
        print("✓ CBAM module loaded successfully")
        print("✓ SimAM module loaded successfully")
        
        # Test CBAM
        cbam = CBAM(256)
        x = torch.randn(1, 256, 32, 32)
        y = cbam(x)
        print(f"  CBAM input shape: {x.shape}, output shape: {y.shape}")
        
        # Test SimAM
        simam = SimAM()
        x = torch.randn(1, 256, 32, 32)
        y = simam(x)
        print(f"  SimAM input shape: {x.shape}, output shape: {y.shape}")
        print()
        return True
    except Exception as e:
        print(f"✗ Error loading attention modules: {e}")
        print()
        return False

def check_dataset(data_path):
    """Kiểm tra xem file cấu hình dataset có tồn tại"""
    print("=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)
    if Path(data_path).exists():
        print(f"✓ Dataset config found: {data_path}")
        with open(data_path, 'r') as f:
            print(f.read())
        print()
        return True
    else:
        print(f"✗ Dataset config NOT found: {data_path}")
        print("  Kiểm tra lại đường dẫn, ví dụ: data/dataset/data.yaml")
        print()
        return False


def resolve_dataset_yaml(data_path):
    """Tự động chuyển data.yaml cloud path sang local path khi cần."""
    data_file = Path(data_path)
    with open(data_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_path = str(cfg.get("path", "")).strip()
    dataset_root = Path(raw_path) if raw_path else data_file.parent
    looks_like_cloud_path = raw_path.startswith("/content") or raw_path.startswith("/kaggle")
    missing_local_path = not dataset_root.exists()

    if looks_like_cloud_path or missing_local_path:
        local_root = data_file.parent.resolve()
        cfg["path"] = str(local_root).replace("\\", "/")
        local_yaml = data_file.with_name(f"{data_file.stem}.local.yaml")
        with open(local_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"⚠ Phát hiện path dataset không hợp lệ cho local: '{raw_path}'")
        print(f"✓ Đã tạo file local dataset config: {local_yaml}")
        print(f"✓ Sử dụng path local: {cfg['path']}")
        return str(local_yaml)

    return str(data_file)


def print_attention_trainable_status(model):
    """In trạng thái requires_grad của tham số attention để kiểm tra freeze."""
    total_attn_params = 0
    trainable_attn_params = 0

    for name, p in model.model.named_parameters():
        lname = name.lower()
        if "cbam" in lname or "simam" in lname:
            total_attn_params += p.numel()
            if p.requires_grad:
                trainable_attn_params += p.numel()

    print(
        f"Attention params trainable: {trainable_attn_params}/{total_attn_params}"
    )

def train():
    """Hàm chính để huấn luyện mô hình"""
    print("=" * 60)
    print("HUẤN LUYỆN YOLO12 VỚI ATTENTION MECHANISMS")
    print("=" * 60)
    print()
    
    # Kiểm tra môi trường
    check_environment()
    
    # Kiểm tra attention modules
    if not test_attention_modules():
        print("WARNING: Không thể load attention modules, tiếp tục huấn luyện...")
    
    # Kiểm tra dataset
    data_path = CONFIG["data"]
    if not check_dataset(data_path):
        print("ERROR: Vui lòng chuẩn bị file cấu hình dataset trước")
        return
    data_path = resolve_dataset_yaml(data_path)

    # Tự động chọn device phù hợp theo môi trường
    requested_device = CONFIG["device"]
    train_device = requested_device
    if isinstance(requested_device, int) and not torch.cuda.is_available():
        print("⚠ CUDA không khả dụng, tự động chuyển sang device='cpu'")
        train_device = "cpu"
    
    # Load model từ file yaml
    model_path = Path(CONFIG["model"])
    print(f"Đang load mô hình từ: {model_path}")
    
    try:
        project_root = Path(__file__).resolve().parent
        candidate_paths = [
            model_path,
            project_root / model_path,
            project_root / "ultralytics" / "ultralytics" / model_path.name,
            project_root / "ultralytics" / model_path.name,
            Path("/kaggle/working/YOLOv12-Attention/ultralytics/ultralytics") / model_path.name,
            Path("/content/drive/MyDrive/Du_An_Ca_Nhan/Computer_Vision/phan_loai_vat_lieu_tai_che/Attention/ultralytics/ultralytics") / model_path.name,
        ]

        resolved_model_path = next((p for p in candidate_paths if Path(p).exists()), None)
        if resolved_model_path is None:
            print(f"ERROR: Không tìm thấy file mô hình: {model_path}")
            print("  Các vị trí đã thử:")
            for p in candidate_paths:
                print(f"   - {p}")
            return

        print(f"✓ Resolved model path: {resolved_model_path}")
        model = YOLO(str(resolved_model_path))
        print(f"✓ Mô hình đã được load")
        print_attention_trainable_status(model)
        print()
        
    except Exception as e:
        print(f"ERROR: Không thể load mô hình: {e}")
        return
    
    # Huấn luyện
    print("=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 60)
    print()
    
    try:
        results = model.train(
            data=data_path,
            epochs=CONFIG["epochs"],
            imgsz=CONFIG["imgsz"],
            batch=CONFIG["batch"],
            device=train_device,
            freeze=CONFIG["freeze"],
            pretrained=CONFIG["pretrained"],
            patience=CONFIG["patience"],
            save=CONFIG["save"],
            plots=CONFIG["plots"],
            optimizer=CONFIG["optimizer"],
            lr0=CONFIG["lr0"],
            lrf=CONFIG["lrf"],
            momentum=CONFIG["momentum"],
            weight_decay=CONFIG["weight_decay"],
            verbose=CONFIG["verbose"],
        )
        
        print("=" * 60)
        print("HUẤN LUYỆN HOÀN THÀNH!")
        print("=" * 60)
        print(f"Kết quả huấn luyện: {results}")
        
        # Lưu mô hình tốt nhất
        print("\nĐang lưu mô hình tốt nhất...")
        model.save("vat_lieu_model_best.pt")
        print("✓ Mô hình đã được lưu: vat_lieu_model_best.pt")
        
    except Exception as e:
        print(f"ERROR: Lỗi trong quá trình huấn luyện: {e}")
        return

def validate():
    """Hàm để kiểm tra mô hình đã huấn luyện"""
    print("=" * 60)
    print("KIỂM TRA MÔ HÌNH")
    print("=" * 60)
    
    model_path = "vat_lieu_model_best.pt"
    data_path = resolve_dataset_yaml(CONFIG["data"])
    
    if not Path(model_path).exists():
        print(f"ERROR: Không tìm thấy mô hình: {model_path}")
        return
    
    model = YOLO(model_path)
    results = model.val(data=data_path)
    print(f"Kết quả kiểm tra: {results}")

def predict(image_path):
    """Hàm sử dụng mô hình để dự đoán"""
    print("=" * 60)
    print("DỰ ĐOÁN TRÊN ẢNH")
    print("=" * 60)
    
    model_path = "vat_lieu_model_best.pt"
    
    if not Path(model_path).exists():
        print(f"ERROR: Không tìm thấy mô hình: {model_path}")
        return
    
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25)
    print(f"Kết quả dự đoán: {results}")
    
    for result in results:
        result.save(filename=f"result_{Path(image_path).name}")
    print(f"✓ Kết quả đã được lưu")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "validate":
            validate()
        elif sys.argv[1] == "predict":
            if len(sys.argv) > 2:
                predict(sys.argv[2])
            else:
                print("Usage: python train.py predict <image_path>")
    else:
        # Mặc định: huấn luyện
        train()
