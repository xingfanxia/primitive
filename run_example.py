#!/usr/bin/env python
"""
Convenience script to run the py_primitive example from the project root.

Usage:
    python run_example.py [input_image] [--size SIZE] [--shapes SHAPES] [--fps FPS] [--mode MODE] [--input-resize RESIZE] [--use-gpu]
    
Options:
    --size SIZE               Set the output image size (default: original image size)
    --shapes SHAPES           Set the number of shapes to generate (default: 100)
    --fps FPS                 Set the animation frames per second (default: 5)
    --mode MODE               Set shape mode: 1=triangle, 2=rectangle, 3=ellipse, 4=circle, 5=rotated_rectangle (default: 1)
    --input-resize RESIZE     Set the size to resize input for processing (default: 256, higher=better quality but slower)
    --use-gpu                 Enable experimental GPU acceleration (may be slower than CPU)
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
                        help='Frames per second for the animation (default: 5)')
    parser.add_argument('--mode', type=int, default=1,
                        help='Shape mode: 1=triangle, 2=rectangle, 3=ellipse, 4=circle, 5=rotated_rectangle (default: 1)')
    parser.add_argument('--input-resize', type=int, default=256,
                        help='Input resize for processing (higher = better quality but slower) (default: 256)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable experimental GPU acceleration (may be slower than CPU)')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the py_primitive directory path
    py_primitive_dir = os.path.join(script_dir, 'py_primitive')
    
    # Ensure root outputs directory exists
    root_outputs_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(root_outputs_dir, exist_ok=True)
    
    # Determine the input image
    input_image = args.input_image
    if input_image is None:
        # Use default image
        input_image = os.path.join(py_primitive_dir, "py_primitive", "examples", "images", "monalisa.png")
    else:
        # Convert relative paths to absolute paths based on current working directory
        if not os.path.isabs(input_image):
            input_image = os.path.abspath(input_image)
        
        # Check if the file exists before proceeding
        if not os.path.exists(input_image):
            print(f"Error: Input file '{input_image}' not found.")
            sys.exit(1)
    
    # Change to the py_primitive directory
    os.chdir(py_primitive_dir)
    
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
    print(f"GPU acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")
    
    # Create a temporary configuration file with custom settings
    output_size_value = args.size
    if output_size_value is None and original_width is not None and original_height is not None:
        # Use exact original dimensions instead of None
        output_size_value = max(original_width, original_height)
        print(f"Setting output_size to match original dimensions: {output_size_value}px")
    
    config_file = os.path.join(py_primitive_dir, 'custom_config.py')
    with open(config_file, 'w') as f:
        f.write(f"""
\"\"\"
Custom configuration for py_primitive.
\"\"\"

config = {{
    "shape_count": {args.shapes},          # Number of shapes to generate
    "shape_mode": {args.mode},             # Shape mode ({shape_mode_name})
    "output_size": {output_size_value},    # Output resolution (original size or specified size)
    "input_resize": {args.input_resize},   # Input resize for processing (higher = better quality but slower)
    "candidates": 100,                     # Number of initial random candidates
    "refinements": 8,                      # Number of refinement steps
    "top_candidates": 5,                   # Number of top candidates to refine
    "batch_size": 50,                      # Balanced batch size for GPU
    "fps": {args.fps},                     # Animation frames per second
    "use_gpu": {str(args.use_gpu).lower().capitalize()} # Enable GPU acceleration
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
    
    # Copy only the generated output files to the root outputs directory
    py_primitive_outputs_dir = os.path.join(py_primitive_dir, 'outputs')
    if os.path.exists(py_primitive_outputs_dir):
        # Get the base name of the input file without extension
        base_filename = os.path.splitext(os.path.basename(input_image))[0]
        
        # Define the file extensions we expect to be generated
        expected_extensions = ['.png', '.svg', '.gif']
        
        # Only copy files matching the base name
        for ext in expected_extensions:
            source_file = os.path.join(py_primitive_outputs_dir, f"{base_filename}{ext}")
            if os.path.exists(source_file):
                dest_path = os.path.join(root_outputs_dir, f"{base_filename}{ext}")
                print(f"Copying {base_filename}{ext} to root outputs directory...")
                shutil.copy2(source_file, dest_path)
        
        print(f"\nOutput files copied to: {root_outputs_dir}")
        
        # For image files, print the actual dimensions by file type
        print("\nOutput file dimensions:")
        
        # Check PNG files
        output_png = os.path.join(root_outputs_dir, f"{base_filename}.png")
        if os.path.exists(output_png):
            try:
                with Image.open(output_png) as img:
                    width, height = img.size
                    print(f"- PNG: {width}x{height} pixels")
            except Exception as e:
                print(f"Error checking PNG dimensions: {e}")
        
        # Check GIF files
        output_gif = os.path.join(root_outputs_dir, f"{base_filename}.gif")
        if os.path.exists(output_gif):
            try:
                with Image.open(output_gif) as img:
                    width, height = img.size
                    print(f"- GIF: {width}x{height} pixels")
            except Exception as e:
                print(f"Error checking GIF dimensions: {e}")
        
        # Report SVG dimensions (can't easily open SVG files with PIL)
        output_svg = os.path.join(root_outputs_dir, f"{base_filename}.svg")
        if os.path.exists(output_svg):
            if output_size_value is not None:
                print(f"- SVG: {output_size_value}x{output_size_value} pixels (based on configuration)")
            elif original_width is not None and original_height is not None:
                print(f"- SVG: {original_width}x{original_height} pixels (based on original image)")
                
        print(f"\nNumber of shapes: {args.shapes}")
        if args.size is not None:
            print(f"Requested output size: {args.size}px")
        elif original_width is not None and original_height is not None:
            print(f"Output size: {max(original_width, original_height)}px (original dimensions)")
        else:
            print(f"Output size: Unknown")
        print(f"Animation FPS: {args.fps}")
        print(f"Shape mode: {args.mode} ({shape_mode_name})")
        print(f"Input resize: {args.input_resize}px (higher = better quality but slower)")
        print(f"GPU acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")

if __name__ == "__main__":
    main() 