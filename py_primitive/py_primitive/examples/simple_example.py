#!/usr/bin/env python3
"""
Simple example using py_primitive library.
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to Python path for easy imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from py_primitive.primitive.model import PrimitiveModel

def main():
    """Run a simple example of primitive image generation."""
    # Check if input image is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_example.py <input_image>")
        print("Example: python simple_example.py ../examples/monalisa.jpg")
        return 1
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create base output path from input filename
    input_filename = os.path.basename(input_path)
    base_name = os.path.splitext(input_filename)[0]
    
    # Configuration
    config = {
        "shape_count": 50,         # Number of shapes to generate
        "shape_mode": 1,           # 1=triangle
        "population_size": 30,     # Population size for DE
        "generations": 10,         # Number of generations for DE
        "input_resize": 256,       # Resize input to this size
    }
    
    print(f"Processing {input_path}...")
    
    # Create model
    start_time = time.time()
    model = PrimitiveModel(input_path, config)
    print(f"Model initialized in {time.time() - start_time:.2f}s")
    
    # Run the algorithm
    print(f"Generating {config['shape_count']} shapes...")
    model.run()
    
    # Save outputs
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"
    gif_path = output_dir / f"{base_name}.gif"
    
    print("Saving outputs...")
    model.save_image(str(png_path))
    model.save_svg(str(svg_path))
    model.save_animation(str(gif_path), frame_count=10)
    
    print("Done!")
    print(f"Results saved to {output_dir} directory.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 