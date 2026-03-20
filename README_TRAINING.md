# 🚀 YOLO12 với Attention Mechanisms - Phân loại Vật liệu Tái chế

## 📦 Tích hợp

✅ **CBAM** (Convolutional Block Attention Module) - Layer 3, 6 trong backbone  
✅ **SimAM** (Simple Attention Module) - Layer 17, 21 trong head  
✅ **YOLO12m** - Model medium cho detection  
✅ **5 lớp** - Giấy, Vải, Nhựa, Kim loại, Thủy tinh  

---

## ⚡ Quick Start

### 1️⃣ Chuẩn bị môi trường
```bash
cd Attention

# Kích hoạt venv
vat_lieu\Scripts\activate  # Windows
# hoặc
source vat_lieu/bin/activate  # Mac/Linux

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2️⃣ Kiểm tra các module Attention
```bash
python test_modules.py
```
**Output mong đợi:**
```
TESTING CBAM (Channel + Spatial Attention)
✓ CBAM test passed!

TESTING SimAM (Simple Attention Module)
✓ SimAM test passed!

✓ ALL TESTS PASSED!
```

### 3️⃣ Chuẩn bị dữ liệu
**Cấu trúc dữ liệu:**
```
data/
├── images/
│   ├── train/      (ảnh training)
│   ├── val/        (ảnh validation)
│   └── test/       (tuỳ chọn)
└── labels/
    ├── train/      (label YOLO format)
    └── val/
```

**Cập nhật `data/vat_lieu.yaml`:**
```yaml
path: D:/path/to/data           # ← Thay đường dẫn thực
train: images/train
val: images/val
nc: 5
names: ['Giấy', 'Vải', 'Nhựa', 'Kim loại', 'Thủy tinh']
```

### 4️⃣ Huấn luyện
```bash
python train.py train
```

### 5️⃣ Kiểm tra & Dự đoán
```bash
# Validation
python train.py validate

# Prediction
python train.py predict path/to/image.jpg
```

---

## 📚 Tài liệu chi tiết

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Hướng dẫn huấn luyện đầy đủ
- **[train.py](train.py)** - Script huấn luyện chính
- **[examples.py](examples.py)** - Ví dụ sử dụng
- **[ultralytics/yolo12m-attn.yaml](ultralytics/yolo12m-attn.yaml)** - Cấu hình mô hình

---

## 📊 Kiến trúc mô hình

```
YOLO12-Attention Architecture
│
├─ Backbone (học feature cơ bản)
│  ├─ P1 Conv (64 channels) 
│  ├─ P2 Conv → C3k2 + CBAM (256)  ← Attention 1
│  ├─ P3 Conv → C3k2 + CBAM (512)  ← Attention 2
│  ├─ P4 Conv → A2C2f (512)
│  └─ P5 Conv → A2C2f (1024)
│
├─ Neck (học multi-scale features)
│  ├─ P5 Upsample → Concat P4
│  │   └─ A2C2f → SimAM  ← Attention 3 (refine small objects)
│  └─ P4 Upsample → Concat P3
│      └─ A2C2f → SimAM  ← Attention 4 (refine reflections)
│
└─ Head (Detection)
   ├─ P3 Detect (small: 8x)
   ├─ P4 Detect (medium: 16x)
   └─ P5 Detect (large: 32x)
```

---

## 🎯 Attention Mechanisms

### CBAM (Convolutional Block Attention Module)
- **Channel Attention**: Tìm channel quan trọng
- **Spatial Attention**: Tìm vùng hình quan trọng
- ✓ Dùng cho: Backbone - học texture & phản xạ vật liệu

### SimAM (Simple Attention Module)
- Tính trọng số từ phương sai spatial
- Nhẹ hơn CBAM, nhanh hơn
- ✓ Dùng cho: Neck - refine vật liệu nhỏ & cạnh

---

## 💻 Yêu cầu hệ thống

```
Python: 3.8+
GPU: NVIDIA (CUDA 11.8+) - khuyến khích
RAM: 8GB+ (16GB+ tốt hơn)
VRAM: 4GB+ (8GB+khuyến khích)
Storage: 20GB+ (dữ liệu + models)
```

---

## 🔧 Tuỳ chỉnh

### Giảm GPU memory
```python
# train.py
CONFIG["batch"] = 8  # Từ 16 → 8
```

### Tăng độ chính xác
```python
CONFIG["epochs"] = 200
CONFIG["patience"] = 30
```

### Tăng tốc độ training
```python
CONFIG["imgsz"] = 416  # Từ 640 → 416
CONFIG["batch"] = 32
```

---

## 📈 Kết quả huấn luyện

Sau khi huấn luyện, kết quả sẽ lưu trong `runs/detect/train*/`:
- `weights/best.pt` - Mô hình tốt nhất
- `results.csv` - Metrics qua các epoch
- `confusion_matrix.png` - Ma trận nhầm lẫn
- `training_plots.png` - Biểu đồ training

---

## 📞 Troubleshooting

### ❌ "CUDA out of memory"
→ Giảm batch size hoặc imgsz trong config

### ❌ "Module not found: cbam"
→ Chạy `pip install -r requirements.txt` lại

### ❌ "No training samples found"
→ Kiểm tra đường dẫn dataset trong `data/vat_lieu.yaml`

### ❌ "Loss không giảm"
→ Tăng learning rate (lr0) hoặc tăng epochs

---

## 🎓 Tham khảo

- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **CBAM Paper**: Woo et al., 2018 - https://arxiv.org/abs/1807.06521
- **SimAM Paper**: Yang et al., 2021 - https://arxiv.org/abs/2102.06171
- **YOLO12**: https://docs.ultralytics.com/models/yolo12

---

## 📝 License

Theo Ultralytics - AGPL-3.0

---

## ✨ Hỗ trợ
- Inspect model: `python examples.py`
- Test attention: `python test_modules.py`
- Training: `python train.py train`

**Chúc bạn huấn luyện thành công! 🚀**
