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
* Features: GPU acceleration, differential evolution, parallel shape evaluation
* See the [Python implementation README](py_primitive/README.md) for more details

## Performance Comparison

The Python GPU implementation offers several advantages over the original Go version:

* **5-10x speedup** for shape evaluation on M-series Macs
* **Improved quality** due to testing more shape variations
* **Better optimization** through differential evolution algorithm
* **Simplified extension** through Python's ecosystem

## Quick Start

### Go Implementation

```bash
cd go_primitive
go build
./primitive -i input.png -o output.png -n 100
```

### Python Implementation

```bash
cd py_primitive
./install.sh
py_primitive -i input.jpg -o output.png -n 100
```

## Examples

![Example](https://www.michaelfogleman.com/static/primitive/examples/16550611738.200.128.4.5.png)

See the individual implementation READMEs for more examples and usage information.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

Original implementation by Michael Fogleman. Python GPU implementation by Xingfan Xia. 