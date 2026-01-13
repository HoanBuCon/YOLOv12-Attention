⚠️ [1] Thư mục gốc của project: `\Attention`  
⚠️ [2] Thư mục gốc của ultralytics: `\Attention\ultralytics`  
⚠️ [3] Thư mục chứa các file cấu hình: `\Attention\ultralytics\ultralytics`  

⚠️ Chạy `pip install --upgrade pip` tại `\attention`  
⚠️ Chạy `pip install -r requirements.txt` tại `\attention`  
⚠️ Chạy `pip install -e .` tại `\attention\ultralytics`  

### 1. Module Attention:
- Đường dẫn: `ultralytics/ultralytics/nn/modules/attention.py`

### 2. Tạo file `cbam.py` và `simam.py` tại `ultralytics/ultralytics/nn/modules`
- Đã tạo, đường dẫn chuẩn 2 file đã tạo:
- cbam.py: `ultralytics/ultralytics/nn/modules/cbam.py`
- simam.py: `ultralytics/ultralytics/nn/modules/simam.py`

### 3. Đăng ký module:
- Đường dẫn: `ultralytics/ultralytics/nn/modules/__init__.py`
- Đường dẫn: `ultralytics/ultralytics/nn/tasks.py`

### 4. Load module, chạy bằng PowerShell:
- `python -c "from ultralytics.nn.modules import CBAM, SimAM; print('OK - Attention modules loaded')"`

### 5. Chạy file `test_attention.py` để kiểm tra module:
- Chạy script bên dưới tại: [3]
- `python test_attention.py`

### 6. Tạo Model YOLOv12 + Attention
- Chạy script bên dưới tại: [3]
- Tìm file YAML cần cấu hình: `dir cfg\models\12`
- Copy model để thêm Attention: `copy cfg\models\12\yolo12.yaml yolo12-attn.yaml`
- Sửa file được copy (đã sửa): Đường dẫn `ultralytics/ultralytics/yolo12-attn.yaml`
- Copy file `yolo12-attn.yaml` và đổi tên thành `yolo12m-attn.yaml` (lách lỗi CLI của Ultralytics)
- Test kiến trúc mới: `yolo benchmark model=yolo12m-attn.yaml task=detect`