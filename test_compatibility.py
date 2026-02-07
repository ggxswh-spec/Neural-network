"""
Test script to verify the neural network modules work correctly on Ubuntu/Debian.
This script doesn't require the training dataset.
"""

import numpy as np
from rknn_conv_operator import create_conv_operator, RKNNDepthwiseConvolutionOperator, ConvolutionConfig

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
        
        from CNN import Conv2D, ReLU, MaxPool2d, Flatten, Dense, Softmax, SimpleCNN
        print("✓ CNN modules imported successfully")
        
        from rknn_conv_operator import RKNNConvolutionOperator
        print("✓ RKNN operator imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cnn_components():
    """Test basic CNN components."""
    print("\nTesting CNN components...")
    try:
        from CNN import Conv2D, ReLU, MaxPool2d, Dense
        
        # Test Conv2D
        conv = Conv2D(1, 8, 3)
        x = np.random.randn(2, 1, 28, 28)
        out = conv.forward(x)
        print(f"✓ Conv2D: input {x.shape} -> output {out.shape}")
        
        # Test ReLU
        relu = ReLU()
        out = relu.forward(out)
        print(f"✓ ReLU: output {out.shape}")
        
        # Test MaxPool2d
        pool = MaxPool2d(2, 2)
        out = pool.forward(out)
        print(f"✓ MaxPool2d: output {out.shape}")
        
        # Test Dense
        dense = Dense(13*13*8, 10)
        x_flat = out.reshape(2, -1)
        out = dense.forward(x_flat)
        print(f"✓ Dense: output {out.shape}")
        
        return True
    except Exception as e:
        print(f"✗ CNN component test failed: {e}")
        return False

def test_rknn_operator():
    """Test RKNN convolution operator."""
    print("\nTesting RKNN operator...")
    try:
        # Test standard convolution
        conv = create_conv_operator(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        
        input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        output = conv.forward(input_data)
        
        print(f"✓ Standard conv: input {input_data.shape} -> output {output.shape}")
        print(f"  Weights dtype: {conv.get_weights().dtype}")
        
        # Test depthwise convolution
        config = ConvolutionConfig(kernel_size=3, stride=1, padding=1)
        dw_conv = RKNNDepthwiseConvolutionOperator(channels=16, config=config)
        
        input_data = np.random.randn(1, 16, 32, 32).astype(np.float32)
        output = dw_conv.forward(input_data)
        
        print(f"✓ Depthwise conv: input {input_data.shape} -> output {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ RKNN operator test failed: {e}")
        return False

def test_full_model():
    """Test the complete CNN model."""
    print("\nTesting complete CNN model...")
    try:
        from CNN import SimpleCNN
        
        model = SimpleCNN()
        x = np.random.randn(2, 1, 28, 28)
        output = model.forward(x)
        
        print(f"✓ SimpleCNN: input {x.shape} -> output {output.shape}")
        print(f"  Output sum: {output.sum():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Full model test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Platform Compatibility Test")
    print("Testing on Ubuntu/Debian systems")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_basic_imports()
    all_passed &= test_cnn_components()
    all_passed &= test_rknn_operator()
    all_passed &= test_full_model()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! The code works correctly on this system.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    print("=" * 60)
