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

def main(custom_config=None):
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
    
    # Configuration - optimized for speed with the new randomized search optimizer
    if custom_config:
        config = custom_config
        print(f"Using custom configuration with output size: {config.get('output_size', 'default')}px")
    else:
        config = {
            "shape_count": 100,          # Number of shapes to generate
            "shape_mode": 1,             # 1=triangle
            "input_resize": 256,         # Input size for processing
            "output_size": 1024,         # Output resolution
            "candidates": 75,            # Number of initial random candidates
            "refinements": 5,            # Number of refinement steps
            "top_candidates": 3,         # Number of top candidates to refine
            "batch_size": 75,            # Process in batches
        }
    
    print(f"Processing {input_path}...")
    
    # Create model
    start_time = time.time()
    model = PrimitiveModel(input_path, config)
    print(f"Model initialized in {time.time() - start_time:.2f}s")
    
    # Run the algorithm
    shape_count = config.get("shape_count", 100)
    print(f"Generating {shape_count} shapes...")
    model.run(num_shapes=shape_count)
    
    # Save outputs
    png_path = output_dir / f"{base_name}.png"
    svg_path = output_dir / f"{base_name}.svg"
    gif_path = output_dir / f"{base_name}.gif"
    
    print("Saving outputs...")
    model.save_image(str(png_path))
    model.save_svg(str(svg_path))
    
    # Use FPS from config if available, otherwise use default (5)
    fps = config.get("fps", 5)
    model.save_animation(str(gif_path), frame_count=10, fps=fps)
    
    print("Done!")
    print(f"Results saved to {output_dir} directory.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 