#!/usr/bin/env python3
"""
Primitive: GPU-accelerated image approximation with geometric primitives.
"""
import argparse
import os
import time
import sys
from typing import Dict, Any

from py_primitive.primitive.model import PrimitiveModel
from py_primitive.config.config import get_config, SHAPE_TYPES

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate primitive images using geometric shapes with GPU acceleration."
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, action='append', default=[], 
                       help='Output image path(s) (can specify multiple times for different formats)')
    parser.add_argument('-n', '--num', type=int, required=True, 
                       help='Number of shapes to generate')
    
    # Shape arguments
    parser.add_argument('-m', '--mode', type=int, default=1,
                       help='Shape mode: 0=combo, 1=triangle, 2=rect, 3=ellipse, 4=circle, '
                            '5=rotatedrect, 6=beziers, 7=rotatedellipse, 8=polygon')
    parser.add_argument('-a', '--alpha', type=int, default=128,
                       help='Alpha value (0-255, use 0 to let algorithm choose)')
    
    # Size arguments
    parser.add_argument('-r', '--resize', type=int, default=256,
                       help='Resize input to this size (max dimension)')
    parser.add_argument('-s', '--size', type=int, default=1024,
                       help='Output image size')
    
    # Performance arguments
    parser.add_argument('-j', '--workers', type=int, default=0,
                       help='Number of parallel workers (default: use all cores)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    # Algorithm parameters
    parser.add_argument('--population', type=int, default=50,
                       help='Population size for differential evolution')
    parser.add_argument('--generations', type=int, default=20,
                       help='Number of generations for differential evolution')
    
    # Output control
    parser.add_argument('--frames', type=int, default=0,
                       help='Number of frames for animation (0 = all shapes)')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments."""
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return False
    
    if args.num < 1:
        print("Error: Number of shapes must be greater than 0.")
        return False
    
    if args.mode not in SHAPE_TYPES:
        print(f"Error: Invalid shape mode {args.mode}. Valid modes: {', '.join([f'{k}={v}' for k, v in SHAPE_TYPES.items()])}")
        return False
    
    if args.alpha < 0 or args.alpha > 255:
        print("Error: Alpha value must be between 0 and 255.")
        return False
    
    if args.resize < 1:
        print("Error: Resize value must be greater than 0.")
        return False
    
    if args.size < 1:
        print("Error: Output size must be greater than 0.")
        return False
    
    if args.population < 10:
        print("Error: Population size should be at least 10.")
        return False
    
    if args.generations < 5:
        print("Error: Number of generations should be at least 5.")
        return False
    
    return True

def create_config_from_args(args) -> Dict[str, Any]:
    """
    Create configuration dictionary from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Configuration dictionary
    """
    return {
        "shape_count": args.num,
        "shape_mode": args.mode,
        "shape_alpha": args.alpha,
        "input_resize": args.resize,
        "output_size": args.size,
        "use_gpu": not args.no_gpu,
        "population_size": args.population,
        "generations": args.generations,
        "num_workers": args.workers if args.workers > 0 else None,
    }

def main():
    """Main function."""
    # Parse and validate arguments
    args = parse_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Print configuration
        print(f"Input: {args.input}")
        print(f"Output: {', '.join(args.output)}")
        print(f"Shapes: {args.num} {SHAPE_TYPES[args.mode]}s")
        print(f"GPU: {'Enabled' if not args.no_gpu else 'Disabled'}")
        
        # Initialize model
        start_time = time.time()
        model = PrimitiveModel(args.input, config)
        print(f"Model initialized in {time.time() - start_time:.2f}s")
        print(f"Image dimensions: {model.width}x{model.height}")
        
        # Run the algorithm
        print(f"Generating {args.num} shapes...")
        model.run()
        
        # Save outputs
        for output_path in args.output:
            ext = os.path.splitext(output_path)[1].lower()
            
            if ext in ['.png', '.jpg', '.jpeg']:
                model.save_image(output_path)
            elif ext == '.svg':
                model.save_svg(output_path)
            elif ext == '.gif':
                model.save_animation(output_path, args.frames if args.frames > 0 else None)
            else:
                print(f"Warning: Unsupported output format for '{output_path}'")
                print(f"Supported formats: .png, .jpg, .svg, .gif")
        
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 