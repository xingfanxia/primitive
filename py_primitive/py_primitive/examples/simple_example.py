#!/usr/bin/env python3
"""
Simple example using py_primitive library.
"""
import os
import sys
import time
import importlib.resources
from pathlib import Path

# Add parent directory to Python path for easy imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from py_primitive.primitive.model import PrimitiveModel

def main():
    """Run a simple example of primitive image generation."""
    # Check if input image is provided
    if len(sys.argv) < 2:
        print("Usage: python -m py_primitive.examples.simple_example <input_image>")
        print("Example: python -m py_primitive.examples.simple_example py_primitive/examples/images/monalisa.png")
        
        # If no image is provided, use a default image
        try:
            # Try to get the path to the included images
            images_path = os.path.join(os.path.dirname(__file__), "images")
            default_image = os.path.join(images_path, "monalisa.png")
            
            if not os.path.exists(default_image):
                print(f"Error: Default image not found at {default_image}")
                return 1
                
            print(f"No image provided, using default image: {default_image}")
            input_path = default_image
        except Exception as e:
            print(f"Error loading default image: {e}")
            return 1
    else:
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
    
    # Configuration - optimized for speed with PSO
    config = {
        "shape_count": 20,          # Number of shapes to generate
        "shape_mode": 1,            # 1=triangle
        "swarm_size": 150,          # Increase swarm size for better exploration
        "iterations": 3,            # Fewer iterations for faster execution
        "input_resize": 128,        # Smaller input size for faster processing
        "batch_size": 150,          # Process entire swarm in one batch if possible
        "cognitive_weight": 1.5,    # Weight for particle's own best position
        "social_weight": 2.0,       # Weight for global best position
        "inertia_weight": 0.6,      # Weight for current velocity
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