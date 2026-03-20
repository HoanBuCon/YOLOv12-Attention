"""
Ví dụ sử dụng YOLO12 với Attention cho phân loại vật liệu tái chế
"""
import torch
from ultralytics import YOLO
from pathlib import Path

# ============= VÍ DỤ 1: Load và thử nghiệm mô hình =============
def example_load_and_test():
    print("\n" + "="*60)
    print("EXAMPLE 1: Load và test mô hình")
    print("="*60)
    
    # Load mô hình từ file yaml (chưa huấn luyện)
    model = YOLO("ultralytics/yolo12m-attn.yaml")
    print("✓ Mô hình đã được load từ YAML")
    
    # Xem cấu trúc mô hình
    print("\nCấu trúc mô hình:")
    print(model.model)
    

# ============= VÍ DỤ 2: Huấn luyện mô hình =============
def example_training():
    print("\n" + "="*60)
    print("EXAMPLE 2: Huấn luyện mô hình")
    print("="*60)
    
    # Cấu hình huấn luyện
    model = YOLO("ultralytics/yolo12m-attn.yaml")
    
    # LƯỚI: Cập nhật đường dẫn dataset thực
    dataset_yaml = "data/vat_lieu.yaml"  # Thay bằng đường dẫn thực
    
    if not Path(dataset_yaml).exists():
        print(f"⚠ Dataset file not found: {dataset_yaml}")
        print("  Vui lòng chuẩn bị dữ liệu trước")
        return
    
    # Huấn luyện
    results = model.train(
        data=dataset_yaml,
        epochs=10,          # Ví dụ: 10 epoch
        imgsz=640,
        batch=16,
        device=0,           # GPU 0, hoặc 'cpu' nếu không có GPU
        patience=5,
        save=True,
        plots=True,
    )
    
    print("\n✓ Huấn luyện hoàn thành!")
    print(f"Kết quả: {results}")


# ============= VÍ DỤ 3: Kiểm tra mô hình đã huấn luyện =============
def example_validation():
    print("\n" + "="*60)
    print("EXAMPLE 3: Kiểm tra (validation) mô hình")
    print("="*60)
    
    # Load mô hình đã huấn luyện
    model = YOLO("runs/detect/train/weights/best.pt")
    
    dataset_yaml = "data/vat_lieu.yaml"
    
    if not Path(dataset_yaml).exists():
        print(f"⚠ Dataset file not found: {dataset_yaml}")
        return
    
    # Kiểm tra
    results = model.val(data=dataset_yaml)
    print(f"\nKết quả validation: {results}")


# ============= VÍ DỤ 4: Dự đoán trên ảnh =============
def example_prediction():
    print("\n" + "="*60)
    print("EXAMPLE 4: Dự đoán trên ảnh")
    print("="*60)
    
    # Load mô hình
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Dự đoán trên ảnh (thay bằng đường dẫn ảnh thực)
    image_path = "path/to/image.jpg"
    
    results = model.predict(
        source=image_path,
        conf=0.5,           # Confidence threshold
        iou=0.45,           # IOU threshold
        save=True,          # Lưu kết quả
        visualize=False,
    )
    
    # Lấy thông tin kết quả
    for result in results:
        print(f"\nKết quả dự đoán cho {result.path}:")
        print(f"  - Số vật thể phát hiện: {len(result.boxes)}")
        
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()
            
            class_names = ['Giấy', 'Vải', 'Nhựa', 'Kim loại', 'Thủy tinh']
            print(f"    - {class_names[cls_id]}: {conf:.2%}")
            print(f"      Coordinates: {box.xyxy.tolist()}")


# ============= VÍ DỤ 5: Export mô hình sang định dạng khác =============
def example_export():
    print("\n" + "="*60)
    print("EXAMPLE 5: Export mô hình")
    print("="*60)
    
    # Load mô hình
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Export sang ONNX
    print("Đang export sang ONNX...")
    model.export(format="onnx")
    print("✓ Export ONNX hoàn thành: best.onnx")
    
    # Export sang TFLite
    print("\nĐang export sang TFLite...")
    model.export(format="tflite")
    print("✓ Export TFLite hoàn thành: best.tflite")
    
    # Export sang other formats
    # Formats: torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle
    
    print("\nCác format khác: torchscript, openvino, coreml, saved_model, tflite, edgetpu")


# ============= VÍ DỤ 6: Sử dụng mô hình trong production =============
def example_production():
    print("\n" + "="*60)
    print("EXAMPLE 6: Sử dụng mô hình trong production")
    print("="*60)
    
    import cv2
    import numpy as np
    
    # Load mô hình
    model = YOLO("best.pt")
    
    # Capture từ camera hoặc video
    cap = cv2.VideoCapture(0)  # 0 = webcam
    
    class_names = ['Giấy', 'Vải', 'Nhựa', 'Kim loại', 'Thủy tinh']
    
    print("Đang capture từ camera... (Nhấn 'q' để exit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dự đoán
        results = model(frame)
        
        # Visualize
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLO12 Attention - Recycled Materials", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============= VÍ DỤ 7: Sử dụng Custom Callbacks =============
def example_with_callbacks():
    print("\n" + "="*60)
    print("EXAMPLE 7: Huấn luyện với Custom Callbacks")
    print("="*60)
    
    from ultralytics.utils.callbacks import Results
    
    model = YOLO("ultralytics/yolo12m-attn.yaml")
    
    # Custom callback
    def on_epoch_end(trainer):
        """Gọi tại cuối mỗi epoch"""
        epoch = trainer.epoch
        metrics = trainer.metrics
        print(f"Epoch {epoch}: Loss={metrics.get('loss', 0):.4f}")
    
    # Huấn luyện (nếu có dataset)
    # results = model.train(
    #     data="data/vat_lieu.yaml",
    #     epochs=100,
    #     callbacks={"on_epoch_end": on_epoch_end},
    # )


# ============= MAIN =============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO12 ATTENTION - RECYCLED MATERIALS CLASSIFICATION")
    print("USAGE EXAMPLES")
    print("="*60)
    
    # Chỉnh sửa các hàm bạn muốn chạy:
    
    try:
        # 1. Load and inspect
        example_load_and_test()
        
        # 2. Training (uncomment nếu có dataset)
        # example_training()
        
        # 3. Validation (uncomment nếu có model)
        # example_validation()
        
        # 4. Prediction (uncomment nếu có image)
        # example_prediction()
        
        # 5. Export (uncomment nếu cần export)
        # example_export()
        
        # 6. Production (uncomment nếu cần chạy real-time)
        # example_production()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Hoàn thành!")
    print("="*60)
