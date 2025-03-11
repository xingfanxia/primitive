#!/usr/bin/env python
"""
Convenience script to run the py_primitive example from the project root.

Usage:
    python run_example.py [input_image] [--size SIZE] [--shapes SHAPES] [--fps FPS] [--mode MODE]
    
Options:
    --size SIZE       Set the output image size (default: original image size)
    --shapes SHAPES   Set the number of shapes to generate (default: 100)
    --fps FPS         Set the animation frames per second (default: 5)
    --mode MODE       Set shape mode: 1=triangle, 2=rectangle, 3=ellipse, 4=circle, 5=rotated_rectangle (default: 1)
"""
import os
import sys
import subprocess
import shutil
import glob
import argparse
from PIL import Image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run py_primitive with custom settings')
    parser.add_argument('input_image', nargs='?', default=None, 
                        help='Path to input image (optional)')
    parser.add_argument('--size', type=int, default=None, 
                        help='Output image size (default: original image size)')
    parser.add_argument('--shapes', type=int, default=100, 
                        help='Number of shapes to generate (default: 100)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Animation frames per second (default: 5)')
    parser.add_argument('--mode', type=int, default=1,
                        help='Shape mode: 1=triangle, 2=rectangle, 3=ellipse, 4=circle, 5=rotated_rectangle (default: 1)')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the py_primitive directory path
    py_primitive_dir = os.path.join(script_dir, 'py_primitive')
    
    # Ensure root outputs directory exists
    root_outputs_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(root_outputs_dir, exist_ok=True)
    
    # Change to the py_primitive directory
    os.chdir(py_primitive_dir)
    
    # Determine the input image
    input_image = args.input_image
    if input_image is None:
        # Use default image
        input_image = os.path.join(py_primitive_dir, "py_primitive", "examples", "images", "monalisa.png")
    
    # Get original image dimensions if needed
    original_width = None
    original_height = None
    if os.path.exists(input_image):
        try:
            with Image.open(input_image) as img:
                original_width, original_height = img.size
                print(f"Original image dimensions: {original_width}x{original_height} pixels")
        except Exception as e:
            print(f"Warning: Could not determine original image dimensions: {e}")
    
    # Map shape mode to a descriptive name
    shape_modes = {
        1: "triangle",
        2: "rectangle",
        3: "ellipse",
        4: "circle",
        5: "rotated_rectangle"
    }
    
    shape_mode_name = shape_modes.get(args.mode, "unknown")
    print(f"Using shape mode: {args.mode} ({shape_mode_name})")
    
    # Create a temporary configuration file with custom settings
    output_size_value = args.size if args.size is not None else "None"
    config_file = os.path.join(py_primitive_dir, 'custom_config.py')
    with open(config_file, 'w') as f:
        f.write(f"""
\"\"\"
Custom configuration for py_primitive.
\"\"\"

config = {{
    "shape_count": {args.shapes},          # Number of shapes to generate
    "shape_mode": {args.mode},             # Shape mode ({shape_mode_name})
    "output_size": {output_size_value},    # Output resolution (None = use original size)
    "input_resize": 256,                   # Input resize for processing (good balance for performance)
    "candidates": 100,                     # Number of initial random candidates
    "refinements": 8,                      # Number of refinement steps
    "top_candidates": 5,                   # Number of top candidates to refine
    "batch_size": 50,                      # Balanced batch size for GPU
    "fps": {args.fps},                     # Animation frames per second
}}
""")
    
    # Construct the command to run
    cmd = [sys.executable, '-c', f"""
import sys
import os
import importlib.util

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Import the custom config
spec = importlib.util.spec_from_file_location("custom_config", "{config_file}")
custom_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_config)

# Import and run the example with our config
from py_primitive.examples.simple_example import main as run_example
sys.argv = [sys.argv[0]]
if "{input_image}" != "None":
    sys.argv.append("{input_image}")
run_example(custom_config=custom_config.config)
"""]
    
    # Run the command
    subprocess.run(cmd)
    
    # Clean up the temporary config file
    os.unlink(config_file)
    
    # Copy the output files to the root outputs directory
    py_primitive_outputs_dir = os.path.join(py_primitive_dir, 'outputs')
    if os.path.exists(py_primitive_outputs_dir):
        for file in glob.glob(os.path.join(py_primitive_outputs_dir, '*')):
            filename = os.path.basename(file)
            dest_path = os.path.join(root_outputs_dir, filename)
            print(f"Copying {filename} to root outputs directory...")
            shutil.copy2(file, dest_path)
        
        print(f"\nAll output files copied to: {root_outputs_dir}")
        
        # For PNG files, print the actual dimensions
        for file in glob.glob(os.path.join(root_outputs_dir, '*.png')):
            try:
                with Image.open(file) as img:
                    width, height = img.size
                    print(f"Final image dimensions: {width}x{height} pixels")
            except Exception as e:
                print(f"Error checking image dimensions: {e}")
                
        print(f"Number of shapes: {args.shapes}")
        if args.size is not None:
            print(f"Requested output size: {args.size}px")
        else:
            print(f"Output size: Original dimensions")
        print(f"Animation FPS: {args.fps}")
        print(f"Shape mode: {args.mode} ({shape_mode_name})")

if __name__ == "__main__":
    main() 