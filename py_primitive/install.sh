#!/bin/bash

# Installation script for py_primitive

set -e  # Exit on error

# Set up virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
fi

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install the package in development mode
echo "Installing py_primitive in development mode..."
pip install -e .

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pillow

# Create a proper test script that properly imports the package
mkdir -p temp
cat > temp/test_import.py << 'EOL'
try:
    import py_primitive.config.config
    import py_primitive.primitive.gpu
    import py_primitive.primitive.shapes
    import py_primitive.primitive.optimizer
    import py_primitive.primitive.model
    import py_primitive.main
    import imageio
    import svgpathtools
    from svgpathtools import Path, Line
    print("All imports successful!")
    exit(0)
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
EOL

# Run import test
echo "Running import test..."
python temp/test_import.py

# Remove temp directory
rm -rf temp

echo ""
echo "Installation complete!"
echo ""
echo "To use py_primitive, run:"
echo "py_primitive -i input.jpg -o output.png -n 100"
echo ""
echo "Or try the example (if you have a test image):"
echo "python -m py_primitive.examples.simple_example path/to/image.jpg"
echo ""

# Try running the command as a test
echo "Testing command installation:"
which py_primitive || echo "WARNING: py_primitive command not found in PATH" 