#!/usr/bin/env python3
import torch

try:
    import mmyolo
    print("✅ Successfully imported mmyolo")
    from yolo_world.engine import YOLO
    print("✅ Successfully imported YOLO-World")
    
    # Initialize a YOLO-World model (without loading weights)
    model = YOLO("yolow-s.pt", task="detection")
    print("✅ Successfully initialized YOLO-World model")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
    else:
        print("⚠️ Running on CPU only")
    
    print("\nYOLO-World installation check completed successfully!")
    print("To run inference, use a pre-trained model weight file and provide an image.")
    
except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Ensure you've activated the virtual environment: source venv/bin/activate")
    print("2. Verify all dependencies are installed correctly")
    print("3. Check for any import errors above and install missing packages")

