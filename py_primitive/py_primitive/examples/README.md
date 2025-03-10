# Primitive Examples

This directory contains example scripts and images for testing and demonstrating the py_primitive package.

## Example Images

The `images/` directory contains sample images you can use to test the py_primitive tool:

- `lenna.png`: Classic test image
- `monalisa.png`: Mona Lisa painting
- `owl.png`: Owl image
- `pyramids.png`: Pyramids image

## Running the Examples

### Using the Command Line Interface

```bash
# Basic example with triangles
py_primitive -i py_primitive/examples/images/monalisa.png -o output.png -n 100

# Rectangle example with SVG output
py_primitive -i py_primitive/examples/images/owl.png -o output.svg -n 100 -m 2

# Create an animation with 20 frames
py_primitive -i py_primitive/examples/images/lenna.png -o animation.gif -n 50 --frames 20
```

### Using the Python API

You can also run the included example script:

```bash
# Run the simple example script
python -m py_primitive.examples.simple_example py_primitive/examples/images/monalisa.png
```

This will create PNG, SVG, and GIF outputs in an `outputs/` directory. 