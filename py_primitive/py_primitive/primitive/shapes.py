"""
Shape definitions and operations for primitive image generation.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
import random
import math

class Shape(ABC):
    """
    Abstract base class for primitive shapes.
    """
    
    def __init__(self, width, height, accelerator):
        """
        Initialize a shape.
        
        Args:
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator for tensor operations
        """
        self.width = width
        self.height = height
        self.gpu = accelerator
        
    @abstractmethod
    def mutate(self):
        """Mutate the shape randomly."""
        pass
    
    @abstractmethod
    def to_tensor(self):
        """Convert the shape to a tensor mask."""
        pass
    
    @abstractmethod
    def to_dict(self):
        """Convert the shape to a dictionary of parameters."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, params, width, height, accelerator):
        """Create a shape from a dictionary of parameters."""
        pass
    
    @abstractmethod
    def crossover(self, other):
        """Create a new shape by combining parameters with another shape."""
        pass
    
    @abstractmethod
    def to_svg(self, color):
        """Convert the shape to SVG format."""
        pass


class Triangle(Shape):
    """Triangle shape defined by three points."""
    
    def __init__(self, width, height, accelerator, points=None):
        """
        Initialize a triangle.
        
        Args:
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            points (list): List of three points [(x1,y1), (x2,y2), (x3,y3)]
        """
        super().__init__(width, height, accelerator)
        
        if points is None:
            # Generate random points
            self.points = [
                (random.uniform(0, width), random.uniform(0, height)),
                (random.uniform(0, width), random.uniform(0, height)),
                (random.uniform(0, width), random.uniform(0, height))
            ]
        else:
            self.points = points
    
    def mutate(self, rate=0.1):
        """
        Mutate the triangle by adjusting its vertex positions.
        
        Args:
            rate (float): Mutation rate determining magnitude of change
        
        Returns:
            Triangle: New mutated triangle
        """
        new_points = []
        for x, y in self.points:
            # Random mutation based on image dimensions and rate
            dx = random.normalvariate(0, self.width * rate)
            dy = random.normalvariate(0, self.height * rate)
            
            new_x = max(0, min(self.width, x + dx))
            new_y = max(0, min(self.height, y + dy))
            
            new_points.append((new_x, new_y))
        
        return Triangle(self.width, self.height, self.gpu, new_points)
    
    def to_tensor(self):
        """
        Convert the triangle to a tensor mask using an optimized algorithm.
        
        Returns:
            torch.Tensor: Binary mask tensor with 1s inside the triangle
        """
        # Get device and dtype from accelerator
        device = self.gpu.device
        dtype = torch.float32
        
        # Extract triangle vertex coordinates and ensure they are float
        x0, y0 = float(self.points[0][0]), float(self.points[0][1])
        x1, y1 = float(self.points[1][0]), float(self.points[1][1])
        x2, y2 = float(self.points[2][0]), float(self.points[2][1])
        
        # Calculate bounding box for the triangle to reduce computation
        x_min = max(0, int(min(x0, x1, x2)))
        y_min = max(0, int(min(y0, y1, y2)))
        x_max = min(self.width, int(max(x0, x1, x2)) + 1)
        y_max = min(self.height, int(max(y0, y1, y2)) + 1)
        
        # If the triangle is outside the image or has no area, return empty mask
        if x_min >= x_max or y_min >= y_max:
            return torch.zeros((self.height, self.width), device=device, dtype=dtype)
        
        # Create coordinate tensors just for the bounding box region
        y_coords = torch.arange(y_min, y_max, device=device, dtype=dtype)
        x_coords = torch.arange(x_min, x_max, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute edge functions for the triangle - this is a more efficient way to test point-in-triangle
        # Each edge function returns positive values on one side of the line and negative on the other
        # A point is inside the triangle if all edge functions are positive (or all negative)
        
        # Edge 1: (x1,y1) -> (x2,y2)
        edge1 = (x1 - x0) * (y_grid - y0) - (y1 - y0) * (x_grid - x0)
        
        # Edge 2: (x2,y2) -> (x0,y0)
        edge2 = (x2 - x1) * (y_grid - y1) - (y2 - y1) * (x_grid - x1)
        
        # Edge 3: (x0,y0) -> (x1,y1)
        edge3 = (x0 - x2) * (y_grid - y2) - (y0 - y2) * (x_grid - x2)
        
        # Check if point is inside triangle (all edges have the same sign)
        # We choose to check if all are >= 0 (clockwise triangle)
        inside = ((edge1 >= 0) & (edge2 >= 0) & (edge3 >= 0)) | ((edge1 <= 0) & (edge2 <= 0) & (edge3 <= 0))
        
        # Create the full mask by embedding the bounding box result
        mask = torch.zeros((self.height, self.width), device=device, dtype=dtype)
        mask[y_min:y_max, x_min:x_max] = inside.float()
        
        return mask
    
    def to_dict(self):
        """
        Convert the triangle to a dictionary of parameters.
        
        Returns:
            dict: Dictionary containing the triangle parameters
        """
        return {
            'type': 'triangle',
            'points': self.points
        }
    
    @classmethod
    def from_dict(cls, params, width, height, accelerator):
        """
        Create a triangle from a dictionary of parameters.
        
        Args:
            params (dict): Dictionary containing the triangle parameters
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            
        Returns:
            Triangle: Created triangle
        """
        points = params['points']
        return cls(width, height, accelerator, points)
    
    def crossover(self, other):
        """
        Create a new triangle by combining parameters with another triangle.
        
        Args:
            other (Triangle): Another triangle to crossover with
            
        Returns:
            Triangle: New triangle created by crossover
        """
        # Randomly select vertices from either parent
        new_points = []
        for i in range(3):
            if random.random() < 0.5:
                new_points.append(self.points[i])
            else:
                new_points.append(other.points[i])
        
        return Triangle(self.width, self.height, self.gpu, new_points)
    
    def to_svg(self, color):
        """
        Convert the triangle to SVG format.
        
        Args:
            color (dict): Color information with r, g, b, a components
            
        Returns:
            str: SVG polygon element
        """
        points_str = " ".join([f"{x},{y}" for x, y in self.points])
        fill = f"rgb({color['r']}, {color['g']}, {color['b']})"
        opacity = color['a'] / 255.0
        
        return f'<polygon points="{points_str}" fill="{fill}" fill-opacity="{opacity}" />'


class Rectangle(Shape):
    """Rectangle shape defined by position, width, and height."""
    
    def __init__(self, width, height, accelerator, params=None):
        """
        Initialize a rectangle.
        
        Args:
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            params (dict): Optional parameters {x, y, w, h}
        """
        super().__init__(width, height, accelerator)
        
        if params is None:
            # Generate random rectangle
            self.x = random.uniform(0, width)
            self.y = random.uniform(0, height)
            self.w = random.uniform(1, width / 2)
            self.h = random.uniform(1, height / 2)
        else:
            self.x = params['x']
            self.y = params['y']
            self.w = params['w']
            self.h = params['h']
    
    def mutate(self, rate=0.1):
        """
        Mutate the rectangle by adjusting its position and dimensions.
        
        Args:
            rate (float): Mutation rate determining magnitude of change
            
        Returns:
            Rectangle: New mutated rectangle
        """
        # Mutate position
        x = max(0, min(self.width, self.x + random.normalvariate(0, self.width * rate)))
        y = max(0, min(self.height, self.y + random.normalvariate(0, self.height * rate)))
        
        # Mutate dimensions
        w = max(1, min(self.width, self.w + random.normalvariate(0, self.width * rate)))
        h = max(1, min(self.height, self.h + random.normalvariate(0, self.height * rate)))
        
        return Rectangle(self.width, self.height, self.gpu, {'x': x, 'y': y, 'w': w, 'h': h})
    
    def to_tensor(self):
        """
        Convert the rectangle to a tensor mask using bounding box optimization.
        
        Returns:
            torch.Tensor: Binary mask tensor with 1s inside the rectangle
        """
        # Get device and dtype from accelerator
        device = self.gpu.device
        dtype = torch.float32
        
        # Convert properties to int for faster processing
        x_min = max(0, int(self.x))
        y_min = max(0, int(self.y))
        x_max = min(self.width, int(self.x + self.w))
        y_max = min(self.height, int(self.y + self.h))
        
        # If the rectangle is outside the image or has no area, return empty mask
        if x_min >= x_max or y_min >= y_max:
            return torch.zeros((self.height, self.width), device=device, dtype=dtype)
        
        # Create an empty mask
        mask = torch.zeros((self.height, self.width), device=device, dtype=dtype)
        
        # Set the rectangle area to 1
        mask[y_min:y_max, x_min:x_max] = 1.0
        
        return mask
    
    def to_dict(self):
        """
        Convert the rectangle to a dictionary of parameters.
        
        Returns:
            dict: Dictionary containing the rectangle parameters
        """
        return {
            'type': 'rectangle',
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h
        }
    
    @classmethod
    def from_dict(cls, params, width, height, accelerator):
        """
        Create a rectangle from a dictionary of parameters.
        
        Args:
            params (dict): Dictionary containing the rectangle parameters
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            
        Returns:
            Rectangle: Created rectangle
        """
        rect_params = {
            'x': params['x'],
            'y': params['y'],
            'w': params['w'],
            'h': params['h']
        }
        return cls(width, height, accelerator, rect_params)
    
    def crossover(self, other):
        """
        Create a new rectangle by combining parameters with another rectangle.
        
        Args:
            other (Rectangle): Another rectangle to crossover with
            
        Returns:
            Rectangle: New rectangle created by crossover
        """
        # Randomly select parameters from either parent
        params = {}
        for key in ['x', 'y', 'w', 'h']:
            if random.random() < 0.5:
                params[key] = getattr(self, key)
            else:
                params[key] = getattr(other, key)
        
        return Rectangle(self.width, self.height, self.gpu, params)
    
    def to_svg(self, color):
        """
        Convert the rectangle to SVG format.
        
        Args:
            color (dict): Color information with r, g, b, a components
            
        Returns:
            str: SVG rect element
        """
        fill = f"rgb({color['r']}, {color['g']}, {color['b']})"
        opacity = color['a'] / 255.0
        
        return f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" fill="{fill}" fill-opacity="{opacity}" />'


class Ellipse(Shape):
    """Ellipse shape defined by center position, x-radius, and y-radius."""
    
    def __init__(self, width, height, accelerator, params=None):
        """
        Initialize an ellipse.
        
        Args:
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            params (dict): Optional parameters {cx, cy, rx, ry}
        """
        super().__init__(width, height, accelerator)
        
        if params is None:
            # Generate random ellipse
            self.cx = random.uniform(0, width)
            self.cy = random.uniform(0, height)
            self.rx = random.uniform(1, width / 4)
            self.ry = random.uniform(1, height / 4)
        else:
            self.cx = params['cx']
            self.cy = params['cy']
            self.rx = params['rx']
            self.ry = params['ry']
    
    def mutate(self, rate=0.1):
        """
        Mutate the ellipse by adjusting its position and radii.
        
        Args:
            rate (float): Mutation rate determining magnitude of change
            
        Returns:
            Ellipse: New mutated ellipse
        """
        # Mutate center
        cx = max(0, min(self.width, self.cx + random.normalvariate(0, self.width * rate)))
        cy = max(0, min(self.height, self.cy + random.normalvariate(0, self.height * rate)))
        
        # Mutate radii
        rx = max(1, min(self.width / 2, self.rx + random.normalvariate(0, self.width * rate)))
        ry = max(1, min(self.height / 2, self.ry + random.normalvariate(0, self.height * rate)))
        
        return Ellipse(self.width, self.height, self.gpu, {'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry})
    
    def to_tensor(self):
        """
        Convert the ellipse to a tensor mask using bounding box optimization.
        
        Returns:
            torch.Tensor: Binary mask tensor with 1s inside the ellipse
        """
        # Get device and dtype from accelerator
        device = self.gpu.device
        dtype = torch.float32
        
        # Convert properties to float
        cx = float(self.cx)
        cy = float(self.cy)
        rx = float(self.rx)
        ry = float(self.ry)
        
        # Calculate bounding box for the ellipse to reduce computation
        x_min = max(0, int(cx - rx))
        y_min = max(0, int(cy - ry))
        x_max = min(self.width, int(cx + rx) + 1)
        y_max = min(self.height, int(cy + ry) + 1)
        
        # If the ellipse is outside the image or has no area, return empty mask
        if x_min >= x_max or y_min >= y_max:
            return torch.zeros((self.height, self.width), device=device, dtype=dtype)
        
        # Create coordinate tensors just for the bounding box region
        y_coords = torch.arange(y_min, y_max, device=device, dtype=dtype)
        x_coords = torch.arange(x_min, x_max, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute normalized squared distance from center
        normalized_x = (x_grid - cx) / rx
        normalized_y = (y_grid - cy) / ry
        dist_squared = normalized_x**2 + normalized_y**2
        
        # Create mask for ellipse (points where normalized distance <= 1)
        ellipse_mask = (dist_squared <= 1.0).float()
        
        # Create the full mask by embedding the bounding box result
        mask = torch.zeros((self.height, self.width), device=device, dtype=dtype)
        mask[y_min:y_max, x_min:x_max] = ellipse_mask
        
        return mask
    
    def to_dict(self):
        """
        Convert the ellipse to a dictionary of parameters.
        
        Returns:
            dict: Dictionary containing the ellipse parameters
        """
        return {
            'type': 'ellipse',
            'cx': self.cx,
            'cy': self.cy,
            'rx': self.rx,
            'ry': self.ry
        }
    
    @classmethod
    def from_dict(cls, params, width, height, accelerator):
        """
        Create an ellipse from a dictionary of parameters.
        
        Args:
            params (dict): Dictionary containing the ellipse parameters
            width (int): Image width
            height (int): Image height
            accelerator (GPUAccelerator): GPU accelerator
            
        Returns:
            Ellipse: Created ellipse
        """
        ellipse_params = {
            'cx': params['cx'],
            'cy': params['cy'],
            'rx': params['rx'],
            'ry': params['ry']
        }
        return cls(width, height, accelerator, ellipse_params)
    
    def crossover(self, other):
        """
        Create a new ellipse by combining parameters with another ellipse.
        
        Args:
            other (Ellipse): Another ellipse to crossover with
            
        Returns:
            Ellipse: New ellipse created by crossover
        """
        # Randomly select parameters from either parent
        params = {}
        for key in ['cx', 'cy', 'rx', 'ry']:
            if random.random() < 0.5:
                params[key] = getattr(self, key)
            else:
                params[key] = getattr(other, key)
        
        return Ellipse(self.width, self.height, self.gpu, params)
    
    def to_svg(self, color):
        """
        Convert the ellipse to SVG format.
        
        Args:
            color (dict): Color information with r, g, b, a components
            
        Returns:
            str: SVG ellipse element
        """
        fill = f"rgb({color['r']}, {color['g']}, {color['b']})"
        opacity = color['a'] / 255.0
        
        return f'<ellipse cx="{self.cx}" cy="{self.cy}" rx="{self.rx}" ry="{self.ry}" fill="{fill}" fill-opacity="{opacity}" />'


def create_random_shape(shape_type, width, height, accelerator):
    """
    Create a random shape of the specified type.
    
    Args:
        shape_type (int): Type of shape to create
        width (int): Image width
        height (int): Image height
        accelerator (GPUAccelerator): GPU accelerator
        
    Returns:
        Shape: Created shape
    """
    if shape_type == 1:
        return Triangle(width, height, accelerator)
    elif shape_type == 2:
        return Rectangle(width, height, accelerator)
    elif shape_type == 3:
        return Ellipse(width, height, accelerator)
    elif shape_type == 4:
        # Circle (special case of ellipse)
        ellipse = Ellipse(width, height, accelerator)
        radius = min(ellipse.rx, ellipse.ry)
        return Ellipse(width, height, accelerator, {'cx': ellipse.cx, 'cy': ellipse.cy, 'rx': radius, 'ry': radius})
    else:
        # Default to triangle
        return Triangle(width, height, accelerator) 