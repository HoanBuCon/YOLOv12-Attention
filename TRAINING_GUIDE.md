# HƯỚNG DẪN HUẤN LUYỆN YOLO12 VỚI ATTENTION MECHANISMS

## 📋 CHUẨN BỊ

### 1. Cảnh báo môi trường Python
```bash
# Cấu hình venv (nếu chưa làm)
cd Attention
python -m venv vat_lieu
# Windows
vat_lieu\Scripts\activate
# Mac/Linux
source vat_lieu/bin/activate
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu
- Tạo folder structure:
  ```
  data/
  ├── images/
  │   ├── train/        (ảnh training)
  │   ├── val/          (ảnh validation)
  │   └── test/         (ảnh test, tuỳ chọn)
  └── labels/
      ├── train/        (label YOLO format)
      └── val/
  ```

- Format label: `<class_id> <x_center> <y_center> <width> <height>`
  - class_id: 0=Giấy, 1=Vải, 2=Nhựa, 3=Kim loại, 4=Thủy tinh
  - Tọa độ chuẩn hóa [0, 1]

### 4. Cập nhật file `data/vat_lieu.yaml`
```yaml
path: /path/to/data        # Thay thế bằng đường dẫn thực
train: images/train
val: images/val
nc: 5
names: ['Giấy', 'Vải', 'Nhựa', 'Kim loại', 'Thủy tinh']
```

### 5. Cập nhật `train.py` nếu cần
- Thay đổi `CONFIG["batch"]` nếu bị lỗi GPU memory
- Điều chỉnh `CONFIG["epochs"]` cho huấn luyện
- Thay đổi learning rate nếu không hội tụ

## 🚀 HỬA LUYỆN

### Huấn luyện cơ bản
```bash
python train.py train
```

### Kiểm tra mô hình
```bash
python train.py validate
```

### Dự đoán trên ảnh
```bash
python train.py predict image.jpg
```

## 📊 MƠ HÌNH KIẾN TRÚC

### YOLO12 + Attention:
```
Backbone:
├── Conv (P1)
├── Conv → C3k2 + CBAM (P2/3)    ← Học texture, biên
├── Conv → C3k2 + CBAM (P3/4)    ← Học phản xạ, kim loại
├── Conv → A2C2f (P4/5)
└── Conv → A2C2f (P5/6)

Neck (FPN):
├── P5 → Upsample + Concat P4 → A2C2f → SimAM (P4)  ← Refine vật liệu nhỏ
├── P4 → Upsample + Concat P3 → A2C2f → SimAM (P3)  ← Refine phản xạ

Head:
├── Detect @ P3 (nhỏ)
├── Detect @ P4 (trung bình)
└── Detect @ P5 (lớn)
```

## 🎯 ATTENTION MECHANISMS

### CBAM (Convolutional Block Attention Module)
- **Channel Attention**: Tìm channel quan trọng
- **Spatial Attention**: Tìm vùng màu hình quan trọng
- Dùng: Layer 3, 6 trong backbone để học texture & kim loại

### SimAM (Simple Attention Module)
- Tính trọng số dựa trên phương sai spatial
- Nhẹ hơn CBAM, tính toán nhanh hơn
- Dùng: Layer 17, 21 trong neck để refine vật liệu nhỏ

## 💾 OUTPUT

Các file sẽ được lưu vào `runs/detect/train*/`:
- `weights/best.pt` - Mô hình tốt nhất
- `weights/last.pt` - Mô hình cuối cùng
- `results.csv` - Kết quả huấn luyện
- `confusion_matrix.png` - Ma trận nhầm lẫn
- `val_batch*.jpg` - Hình ảnh validation

## ⚙️ TUYNING SIÊU THAM SỐ

### Nếu lỗi GPU memory:
```python
CONFIG["batch"] = 8  # Giảm batch size
```

### Nếu underfitting (loss cao):
```python
CONFIG["epochs"] = 200  # Tăng epoch
CONFIG["lr0"] = 0.02    # Tăng learning rate
```

### Nếu overfitting (val loss > train loss):
```python
CONFIG["patience"] = 10  # Early stopping sớm hơn
CONFIG["scale"] = 0.7    # Tăng augmentation
```

## 🧪 KIỂM TRA VÀ DỰ ĐOÁN

```python
from ultralytics import YOLO

# Load mô hình
model = YOLO("vat_lieu_model_best.pt")

# Dự đoán
results = model.predict(source="image.jpg", conf=0.5)

# Visualize
for r in results:
    print(r.boxes)  # Bounding boxes
    print(r.probs)  # Probabilities (nếu classification)
```

## 📝 GHI CHÚ

1. **Cân bằng dữ liệu**: Đảm bảo mỗi lớp có số lượng ảnh gần nhau
2. **Augmentation**: Script đã có augmentation, có thể điều chỉnh trong CONFIG
3. **Validation**: Dùng 10% dữ liệu làm validation (có thể thay đổi)
4. **Export**: Sau khi huấn luyện, có thể export sang ONNX, TFLite, v.v.

## 🔗 TÀI LIỆU THAM KHẢO

- Ultralytics YOLO: https://docs.ultralytics.com/
- CBAM Paper: https://arxiv.org/abs/1807.06521
- SimAM Paper: https://arxiv.org/abs/2102.06171
