# Primitive Pictures

Reproducing images with geometric primitives.

This repository contains two implementations of the Primitive algorithm:

1. **Original Go Implementation** - The classic CPU-based implementation
2. **Python GPU Implementation** - A modern GPU-accelerated version using PyTorch with Metal Performance Shaders

## Implementations

### Go Implementation (Original)

The original implementation by Michael Fogleman is written in Go and uses hill climbing optimization to find optimal shapes.

* Location: [go_primitive/](go_primitive/)
* Features: CPU-based, hill climbing optimization, multi-core support
* See the [Go implementation README](go_primitive/README.md) for more details

### Python GPU Implementation (New)

A modern reimplementation that uses GPU acceleration on M-series Macs for significantly improved performance.

* Location: [py_primitive/](py_primitive/)
* Features: GPU acceleration, advanced optimization, parallel shape evaluation, animation control
* See the [Python implementation README](py_primitive/README.md) for more details

## Performance Comparison

The Python GPU implementation offers several advantages over the original Go version:

* **5-10x speedup** for shape evaluation on M-series Macs
* **Improved quality** due to testing more shape variations
* **Better optimization** through advanced optimization algorithms
* **Simplified extension** through Python's ecosystem

## Quick Start

### Go Implementation

```bash
cd go_primitive
go build
./primitive -i input.png -o output.png -n 100
```

### Python Implementation

#### Full CLI

```bash
cd py_primitive
./install.sh
py_primitive -i input.jpg -o output.png -n 100
```

#### Convenient Wrapper Script

Use the wrapper script for a simpler interface that handles common defaults:

```bash
# From project root
python run_example.py [input_image] [options]

# Examples:
# Generate 50 triangles with original image size
python run_example.py --shapes 50

# Use rectangles instead of triangles
python run_example.py --shapes 50 --mode 2

# Create a slow animation (3 FPS)
python run_example.py --shapes 50 --fps 3

# Create high-resolution output (2048px)
python run_example.py --shapes 50 --size 2048
```

## Supported Shape Types

Both implementations support multiple primitive shape types:

1. **Triangles (mode 1)** - Default, excellent balance of simplicity and expressiveness
2. **Rectangles (mode 2)** - Good for architectural images with straight lines
3. **Ellipses (mode 3)** - Effective for organic forms and portraits
4. **Circles (mode 4)** - Creates a more abstract style
5. **Rotated Rectangles (mode 5)** - Useful for diagonal features

## Animation Control

The Python implementation supports controlling animation playback speed:

```bash
# Create a faster animation (10 FPS)
py_primitive -i input.jpg -o animation.gif -n 100 --frames 50 --fps 10

# Create a slower, more detailed animation (3 FPS)
py_primitive -i input.jpg -o animation.gif -n 100 --frames 50 --fps 3
```

## Examples

![Example](https://www.michaelfogleman.com/static/primitive/examples/16550611738.200.128.4.5.png)

See the individual implementation READMEs for more examples and usage information.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

Original implementation by Michael Fogleman. Python GPU implementation by Xingfan Xia. 