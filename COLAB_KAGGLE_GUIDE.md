# HƯỚNG DẪN HUẤ LUYỆN TRÊN KAGGLE VÀ GOOGLE COLAB

## 🐱 KAGGLE NOTEBOOK

### 1️⃣ Chuẩn bị trên Kaggle

**Option A: Upload code lên Kaggle Dataset**
1. Vào https://www.kaggle.com/settings/datasets
2. Click "Create New Dataset"
3. Upload folder `ultralytics/` chứa code + `cbam.py` + `simam.py`
4. Upload dữ liệu training (ảnh + labels YOLO format)

**Option B: Sử dụng Kaggle API**
```bash
# Local machine
kaggle datasets create -p /path/to/ultralytics
kaggle datasets create -p /path/to/data
```

### 2️⃣ Kaggle Notebook Code

```python
# Cell 1: Install packages
!pip install -q ultralytics opencv-python pyyaml matplotlib

# Cell 2: Thêm paths
import sys
from pathlib import Path

# Dataset inputs của Kaggle
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")

# Thêm ultralytics folder vào path
sys.path.insert(0, str(INPUT / "ultralytics" / "ultralytics"))  # Tùy vào cấu trúc dataset

# Cell 3: Test import
try:
    from ultralytics.nn.modules import CBAM, SimAM
    print("✓ Modules loaded!")
except:
    print("Check dataset structure")

# Cell 4: Load và train
from ultralytics import YOLO

model = YOLO(str(INPUT / "ultralytics" / "ultralytics" / "yolo12m-attn.yaml"))

results = model.train(
    data=str(INPUT / "vat-lieu-data" / "data.yaml"),
    epochs=100,
    batch=16,
    device=0,
    imgsz=640,
)

# Cell 5: Save results
model.save(OUTPUT / "best_model.pt")
```

---

## 🔗 GOOGLE COLAB

### 1️⃣ Setup Code

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 2: Install packages
!pip install -q ultralytics opencv-python pyyaml

# Cell 3: Thêm paths
import sys
sys.path.insert(0, "/content/drive/MyDrive/path/to/ultralytics")

# Cell 4: Clone từ GitHub (nếu cần)
# !git clone https://github.com/YOUR_USERNAME/repo.git
# %cd /content/repo

# Cell 5: Test import
from ultralytics.nn.modules import CBAM, SimAM
print("✓ Ready to train!")

# Cell 6: Training
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/.../ultralytics/yolo12m-attn.yaml")

results = model.train(
    data="/content/drive/MyDrive/.../data/vat_lieu.yaml",
    epochs=100,
    batch=16,
    device=0,
)

# Cell 7: Download results
from google.colab import files
files.download("/content/drive/MyDrive/.../runs/detect/train/weights/best.pt")
```

---

## 📦 CÁCH TỐT NHẤT: Publish lên GitHub + Clone

### Step 1: Push code lên GitHub
```bash
cd Attention
git init
git add .
git commit -m "Add YOLO12 with Attention"
git push -u origin main
```

### Step 2: Colab
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Cài editable mode
!pip install -e ultralytics/

# Training
from ultralytics import YOLO
model = YOLO("ultralytics/yolo12m-attn.yaml")
results = model.train(data="data/vat_lieu.yaml", epochs=100, batch=16)
```

### Step 3: Kaggle
```python
# Upload GitHub link hoặc push dataset
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Cài đặt tương tự
```

---

## ⚙️ CẤU HÌNH DATASET

### Colab
```yaml
# Google Drive structure
MyDrive/
├── Du_An_Ca_Nhan/
│   └── Computer_Vision/phan_loai_vat_lieu_tai_che/Attention/
│       ├── ultralytics/  ← code
│       ├── data/
│       │   └── vat_lieu.yaml
│       └── images/
│           ├── train/
│           └── val/

# vat_lieu.yaml
path: /content/drive/MyDrive/Du_An_Ca_Nhan/.../data
train: ../images/train
val: ../images/val
```

### Kaggle
```yaml
# Kaggle datasets
/kaggle/input/
├── ultralytics-code/
├── vat-lieu-data/
│   ├── data.yaml
│   ├── images/train/
│   └── images/val/

# data.yaml
path: /kaggle/input/vat-lieu-data
train: images/train
val: images/val
```

---

## 🔧 FIX COMMON ERRORS

### ❌ "ModuleNotFoundError: No module named 'ultralytics.nn.modules.cbam'"

**Colab:**
```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/.../ultralytics")
sys.path.insert(0, "/content/drive/MyDrive/.../ultralytics/ultralytics")
```

**Kaggle:**
```python
import sys
sys.path.insert(0, "/kaggle/input/ultralytics-code/ultralytics")
sys.path.insert(0, "/kaggle/input/ultralytics-code/ultralytics/ultralytics")
```

### ❌ "CBAM and SimAM files not found"

**Giải pháp:**
- Đảm bảo `cbam.py` + `simam.py` trong `ultralytics/ultralytics/nn/modules/`
- Cập nhật `__init__.py` để import

---

## 📊 BENCHMARK

| Platform | GPU Memory | Training Speed | Notes |
|----------|-----------|-----------------|-------|
| **Local** | 8GB+ | ⚡ Nhanh nhất | GPU dedicated |
| **Colab** | 15GB | ⚡ Khá nhanh | Free, limited time |
| **Kaggle** | 16GB | ⚡ Nhanh | Free, 9hr limit |

---

## 💡 TIPS

1. **Backup Model**: Lưu model thường xuyên
   ```python
   model.save("backup.pt")
   ```

2. **Check GPU Memory**:
   - Colab: `!nvidia-smi`
   - Kaggle: `!nvidia-smi`

3. **Early Stopping**:
   ```python
   model.train(..., patience=20)  # Stop nếu không cải thiện
   ```

4. **Reduce Batch** nếu Out of Memory:
   ```python
   batch=8  # Từ 16 → 8
   ```

5. **Export sau training**:
   ```python
   model.export(format="onnx")
   model.export(format="tflite")
   ```
