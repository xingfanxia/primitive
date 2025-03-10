#!/bin/bash

# Installation script for py_primitive

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

# Run import test
echo "Running import test..."
python tests/test_import.py

echo ""
echo "Installation complete!"
echo ""
echo "To use py_primitive, run:"
echo "py_primitive -i input.jpg -o output.png -n 100"
echo ""
echo "Or try the example (if you have a test image):"
echo "python examples/simple_example.py path/to/image.jpg"
echo "" 