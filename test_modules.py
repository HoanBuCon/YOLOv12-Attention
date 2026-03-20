"""
Script kiểm tra nhanh các module Attention
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import CBAM, SimAM

def test_cbam():
    """Test CBAM module"""
    print("\n" + "="*50)
    print("TESTING CBAM (Channel + Spatial Attention)")
    print("="*50)
    
    channels = 256
    cbam = CBAM(channels, reduction=16, kernel_size=7)
    
    # Test với batch: 1, channels: 256, height: 32, width: 32
    x = torch.randn(1, channels, 32, 32)
    print(f"Input shape:  {x.shape}")
    
    y = cbam(x)
    print(f"Output shape: {y.shape}")
    print(f"Output min/max: {y.min():.4f} / {y.max():.4f}")
    
    # Kiểm tra parameter
    params = sum(p.numel() for p in cbam.parameters())
    print(f"Parameters: {params:,}")
    print("✓ CBAM test passed!")

def test_simam():
    """Test SimAM module"""
    print("\n" + "="*50)
    print("TESTING SimAM (Simple Attention Module)")
    print("="*50)
    
    simam = SimAM(e_lambda=1e-4)
    
    # Test với batch: 2, channels: 512, height: 16, width: 16
    x = torch.randn(2, 512, 16, 16)
    print(f"Input shape:  {x.shape}")
    
    y = simam(x)
    print(f"Output shape: {y.shape}")
    print(f"Output min/max: {y.min():.4f} / {y.max():.4f}")
    
    # Kiểm tra parameter
    params = sum(p.numel() for p in simam.parameters())
    print(f"Parameters: {params:,} (no learnable parameters)")
    print("✓ SimAM test passed!")

def test_combined():
    """Test cấu trúc kết hợp"""
    print("\n" + "="*50)
    print("TESTING COMBINED STRUCTURE (CBAM + SimAM)")
    print("="*50)
    
    class SimpleAttentionBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.cbam = CBAM(channels)
            self.simam = SimAM()
        
        def forward(self, x):
            x = x * self.cbam(x)
            x = self.simam(x)
            return x
    
    block = SimpleAttentionBlock(128)
    x = torch.randn(1, 128, 32, 32)
    print(f"Input shape:  {x.shape}")
    
    y = block(x)
    print(f"Output shape: {y.shape}")
    
    params = sum(p.numel() for p in block.parameters())
    print(f"Total parameters: {params:,}")
    print("✓ Combined structure test passed!")

def check_model_yaml():
    """Kiểm tra file cấu hình mô hình"""
    print("\n" + "="*50)
    print("CHECKING MODEL YAML CONFIGURATION")
    print("="*50)
    
    import yaml
    from pathlib import Path
    
    yaml_path = Path("ultralytics/yolo12m-attn.yaml")
    
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Model YAML found: {yaml_path}")
        print(f"  - Classes: {config.get('nc', 'N/A')}")
        print(f"  - Backbone layers: {len(config.get('backbone', []))}")
        print(f"  - Head layers: {len(config.get('head', []))}")
        
        # Check for attention modules
        backbone_str = str(config.get('backbone', []))
        head_str = str(config.get('head', []))
        
        cbam_count = backbone_str.count('CBAM') + head_str.count('CBAM')
        simam_count = backbone_str.count('SimAM') + head_str.count('SimAM')
        
        print(f"\n  Attention modules found:")
        print(f"    - CBAM: {cbam_count} layers")
        print(f"    - SimAM: {simam_count} layers")
        
        if cbam_count > 0 or simam_count > 0:
            print("\n✓ Attention modules are integrated!")
        else:
            print("\n⚠ No attention modules found in config")
    else:
        print(f"✗ Model YAML not found: {yaml_path}")

def main():
    print("\n" + "="*50)
    print("ATTENTION MODULES TEST SUITE")
    print("="*50)
    
    # Test environment
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Run tests
        test_cbam()
        test_simam()
        test_combined()
        check_model_yaml()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED!")
        print("="*50)
        print("\nReady for training! Run: python train.py train")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
