# Primitive Pictures - GPU Accelerated Python Implementation

Reproducing images with geometric primitives, accelerated by GPU on M-series Macs.

## Overview

This is a Python reimplementation of the [original Primitive project](https://github.com/fogleman/primitive) by Michael Fogleman, with significant enhancements:

1. **GPU Acceleration**: Uses PyTorch with Metal Performance Shaders (MPS) for GPU acceleration on M-series Macs
2. **Optimized Shape Search**: Uses advanced optimization algorithms for better shape finding
3. **Batch Processing**: Evaluates many shapes simultaneously in parallel on the GPU
4. **Python Ecosystem**: Takes advantage of the Python ecosystem for image processing and visualization

The algorithm approximates images by finding optimal geometric shapes (triangles, rectangles, ellipses, etc.) to add one at a time, with each shape improving the overall approximation.

## Installation

### Requirements

- Python 3.7 or later
- PyTorch 1.12 or later
- M-series Mac (for GPU acceleration) or any computer with PyTorch support

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/primitive.git
cd primitive

# Install from source
pip install -e py_primitive
```

## Usage

### Command Line

```bash
py_primitive -i input.jpg -o output.png -n 100 -m 1
```

### Quick Start with run_example.py

For convenience, you can use the wrapper script that has some sensible defaults:

```bash
python run_example.py [input_image] [--size SIZE] [--shapes SHAPES] [--fps FPS]
```

This script will:
- Use the default Mona Lisa image if none is provided
- Auto-detect and preserve original image dimensions by default
- Generate triangles (the most effective primitive shape)
- Save outputs to the `/outputs` directory

### Parameters

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | n/a | input file (required) |
| `-o, --output` | n/a | output file (required, can be specified multiple times) |
| `-n, --num` | n/a | number of shapes (required) |
| `-m, --mode` | 1 | 0=combo, 1=triangle, 2=rectangle, 3=ellipse, 4=circle, 5=rotated_rectangle |
| `-a, --alpha` | 128 | alpha value (0-255, use 0 to let algorithm choose) |
| `-r, --resize` | 256 | resize input image to this size before processing |
| `-s, --size` | 1024 | output image size |
| `-j, --workers` | 0 | number of parallel workers (default uses all cores) |
| `--no-gpu` | false | disable GPU acceleration |
| `--population` | 50 | population size for optimization algorithm |
| `--generations` | 20 | number of generations for optimization algorithm |
| `--frames` | 0 | number of frames for animation (0 = all shapes) |
| `--fps` | 5 | frames per second for animation playback |
| `-v, --verbose` | false | enable verbose output |

### Output Formats

- PNG/JPG: Raster output
- SVG: Vector output
- GIF: Animated output showing shapes being added

### Examples

```bash
# Basic triangle example, 100 shapes
py_primitive -i input.jpg -o output.png -n 100

# Rectangle example with 200 shapes, save both PNG and SVG
py_primitive -i input.jpg -o output.png -o output.svg -n 200 -m 2

# Create an animation with 50 frames
py_primitive -i input.jpg -o animation.gif -n 100 --frames 50

# Use ellipses with higher population and generations for better results
py_primitive -i input.jpg -o output.png -n 50 -m 3 --population 100 --generations 30

# Create a slower animation with 3 FPS
py_primitive -i input.jpg -o animation.gif -n 100 --frames 50 --fps 3
```

## Shape Types

The following shape types are supported:

1. **Triangle (mode 1)**: Three-point polygons, excellent for most image approximations
2. **Rectangle (mode 2)**: Axis-aligned rectangles
3. **Ellipse (mode 3)**: Axis-aligned ellipses
4. **Circle (mode 4)**: Perfect circles
5. **Rotated Rectangle (mode 5)**: Rectangles with arbitrary rotation

Each shape type offers different characteristics:
- Triangles provide excellent coverage and detail with minimal shapes
- Rectangles work well for architectural images with straight lines
- Ellipses and circles excel at organic forms and portraits
- Rotated rectangles can capture diagonal features effectively

## Performance

The GPU-accelerated implementation offers significantly improved performance over the original Go version, especially for larger images and complex shapes. On M-series Macs, you can expect:

- **5-10x speedup** for shape evaluation
- **Improved quality** due to testing more shape variations
- **Better CPU utilization** during processing

### Performance Optimizations

This implementation includes several optimizations for speed:

1. **Batched GPU Processing**: Shapes are evaluated in parallel batches for maximum GPU utilization
2. **Early Stopping**: The optimization algorithm stops early when no improvements are detected
3. **Tensor Caching**: Common tensor operations are cached to reduce redundant calculations
4. **Memory Management**: Periodic GPU memory clearing prevents out-of-memory issues
5. **Progress Tracking**: Real-time progress reporting with time estimates

### Performance Tuning

You can adjust these parameters for optimal performance on your hardware:

```bash
# Faster execution with smaller resolution and fewer shapes/generations
py_primitive -i input.jpg -o output.png -n 50 -r 128 --population 20 --generations 5

# Higher quality with more shapes and generations (slower)
py_primitive -i input.jpg -o output.png -n 200 -r 256 --population 50 --generations 20
```

## How It Works

The core algorithm works as follows:

1. Start with a blank canvas (or average color of target image)
2. For each iteration:
   - Initialize a population of random shapes
   - Evaluate each shape in parallel on the GPU
   - Run optimization to find the optimal shape
   - Add the best shape to the canvas
3. Repeat until the desired number of shapes is reached

## Animation Control

When creating GIF animations, you can control:

- **Frame Count**: Set with `--frames` (0 means one frame per shape)
- **Playback Speed**: Set with `--fps` (default is 5 FPS for smooth but detailed viewing)

## Python API

You can also use the library programmatically:

```python
from py_primitive.primitive.model import PrimitiveModel

# Create model
model = PrimitiveModel("input.jpg")

# Run for 100 shapes
model.run(100)

# Save outputs
model.save_image("output.png")
model.save_svg("output.svg")
model.save_animation("animation.gif", frame_count=20, fps=5)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project is a Python reimplementation of the original [Primitive](https://github.com/fogleman/primitive) by Michael Fogleman. 