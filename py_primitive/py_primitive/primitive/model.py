"""
Main model for generating primitive images.
"""
import numpy as np
import PIL.Image
from PIL import Image
import torch
import time
import os
import math
import random
import imageio
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
from typing import List, Dict, Any, Tuple

from py_primitive.primitive.shapes import create_random_shape, Triangle, Rectangle, Ellipse, Shape
from py_primitive.config.config import get_config, SHAPE_TYPES
from py_primitive.primitive.optimizer import RandomizedShapeOptimizer
from py_primitive.primitive.accelerator import GPUAccelerator, CPUAccelerator

class PrimitiveModel:
    """
    Main model for generating primitive images using GPU acceleration.
    """
    
    def __init__(self, target_image_path, config=None):
        """
        Initialize the model.
        
        Args:
            target_image_path (str): Path to the target image
            config (dict, optional): Configuration parameters
        """
        # Load config settings or use defaults
        self.config = get_config(config)

        # Initialize device and accelerator
        self.use_gpu = self.config.get("use_gpu", True) and torch.cuda.is_available()
        if self.use_gpu:
            self.accelerator = GPUAccelerator()
            print("Using GPU for acceleration")
        else:
            self.accelerator = CPUAccelerator()
            print("Using CPU for computation")

        # Target-related attributes
        self.target_image_path = target_image_path
        self.target_image_tensor = None
        self.current_image_tensor = None
        self.width = None
        self.height = None
        self.bg_color = None

        # Initialize target image
        self._load_and_resize_image(target_image_path)
        self.bg_color = self._compute_background_color()
        self.current_image_tensor = self._create_background_image(self.bg_color)

        # Initialize optimizer
        self.optimizer = RandomizedShapeOptimizer(
            self.config, 
            self.accelerator, 
            self.width, 
            self.height
        )

        # History tracking
        self.score_history = []
        self.shape_history = []
        self.color_history = []
        self.score = float('inf')

        # Initialize with first score calculation
        self.score = self._compute_score()
        self.score_history.append(self.score)

        # Initialize model timings
        self.init_time = time.time()
        print(f"Model initialized in {time.time() - self.init_time:.2f} seconds")
        
    def _load_and_resize_image(self, image_path):
        """
        Load and resize the target image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            PIL.Image: Resized image
        """
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Resize if needed
        if self.config["input_resize"] > 0:
            # Calculate new dimensions while preserving aspect ratio
            w, h = img.size
            ratio = min(self.config["input_resize"] / w, self.config["input_resize"] / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        self.target_image_tensor = self.accelerator.to_tensor(np.array(img) / 255.0)
        self.width, self.height = img.size
        
        return img
        
    def _compute_background_color(self):
        """
        Compute the average color of the target image.
        
        Returns:
            np.ndarray: Background color as RGB values in [0,1]
        """
        # Compute average color across all pixels
        avg_color = torch.mean(self.target_image_tensor, dim=(1, 2))
        return self.accelerator.to_numpy(avg_color)
    
    def _create_background_image(self, bg_color):
        """
        Create a background image filled with the specified color.
        
        Args:
            bg_color: Background color as RGB values in [0,1]
            
        Returns:
            torch.Tensor: Background image tensor
        """
        # Convert the background color to a tensor with the correct shape for broadcasting
        bg_tensor = self.accelerator.to_tensor(bg_color).view(-1, 1, 1)
        
        # Create a tensor of the same shape as target filled with the background color
        return bg_tensor.expand_as(self.target_image_tensor)
    
    def _compute_score(self):
        """
        Compute the current score (difference between target and current).
        
        Returns:
            float: Score as mean squared error
        """
        return self.accelerator.compute_image_difference(self.target_image_tensor, self.current_image_tensor)
    
    def step(self, shape_type=None, alpha=None):
        """
        Add one shape to the current image.
        
        Args:
            shape_type (int): Type of shape to add (None uses config)
            alpha (int): Alpha value for the shape (None uses config)
            
        Returns:
            float: New score after adding the shape
        """
        # Use config values if not specified
        if shape_type is None:
            shape_type = self.config["shape_mode"]
        if alpha is None:
            alpha = self.config["shape_alpha"]
        
        start_time = time.time()
        
        print(f"Finding best {SHAPE_TYPES[shape_type]} shape...")
        
        # Find the best shape to add
        shape, color, score = self.optimizer.find_best_shape(
            shape_type,
            self.target_image_tensor,
            self.current_image_tensor,
            alpha
        )
        
        # Add the shape to the current image
        self._add_shape(shape, color)
        
        elapsed = time.time() - start_time
        shapes_per_sec = 1.0 / elapsed
        
        print(f"Added {SHAPE_TYPES[shape_type]}, score={score:.6f}, time={elapsed:.2f}s, shapes/sec={shapes_per_sec:.2f}")
        
        # Clear GPU cache periodically
        if hasattr(self.accelerator, '_clear_cache'):
            self.accelerator._clear_cache()
        
        return score
    
    def _add_shape(self, shape, color):
        """
        Add a shape to the current image.
        
        Args:
            shape (Shape): Shape to add
            color (dict): Color information
        """
        # Convert shape to mask
        mask = shape.to_tensor()
        
        # Convert RGB components to tensor
        rgb_tensor = torch.tensor([color['r'], color['g'], color['b']], device=self.accelerator.device) / 255.0
        alpha_value = color['a'] / 255.0
        
        # Expand mask and rgb tensor to match image dimensions for proper broadcasting
        mask_expanded = mask.unsqueeze(0).expand_as(self.current_image_tensor)
        rgb_expanded = rgb_tensor.view(-1, 1, 1).expand_as(self.current_image_tensor)
        
        # Apply alpha blending with proper broadcasting
        self.current_image_tensor = self.current_image_tensor * (1 - mask_expanded * alpha_value) + rgb_expanded * mask_expanded * alpha_value
        
        # Update state
        self.shape_history.append(shape)
        self.color_history.append(color)
        self.score_history.append(self._compute_score())
    
    def run(self, num_shapes=None):
        """
        Run the algorithm to add multiple shapes.
        
        Args:
            num_shapes (int): Number of shapes to add (None uses config)
            
        Returns:
            List[float]: Scores after adding each shape
        """
        if num_shapes is None:
            num_shapes = self.config["shape_count"]
        
        start_time = time.time()
        
        # Pre-initialize variables for optimization
        shape_type = self.config.get("shape_mode", 1)
        alpha = self.config.get("shape_alpha", 128)
        
        print(f"Processing {num_shapes} shapes with {SHAPE_TYPES[shape_type]}s")
        print(f"Using acceleration: {self.accelerator.device}")
        
        # Instead of processing shapes one by one, process them in small batches
        # This allows for better parallelization between shapes
        batch_size = min(5, num_shapes)  # Process 5 shapes at a time
        for batch_start in range(0, num_shapes, batch_size):
            batch_end = min(batch_start + batch_size, num_shapes)
            batch_count = batch_end - batch_start
            
            print(f"Processing shapes {batch_start+1}-{batch_end} of {num_shapes}")
            
            for i in range(batch_count):
                shape_num = batch_start + i + 1
                print(f"Shape {shape_num}/{num_shapes}")
                self.step(shape_type, alpha)
            
            # Provide progress update
            elapsed = time.time() - start_time
            shapes_done = batch_end
            time_per_shape = elapsed / shapes_done
            remaining_shapes = num_shapes - shapes_done
            est_remaining_time = remaining_shapes * time_per_shape
            print(f"Progress: {shapes_done}/{num_shapes} shapes, " 
                  f"elapsed: {elapsed:.1f}s, "
                  f"est. remaining: {est_remaining_time:.1f}s, "
                  f"rate: {shapes_done/elapsed:.2f} shapes/sec")
            
            # Force GPU memory cleanup after each batch
            if hasattr(self.accelerator, '_clear_cache'):
                self.accelerator._clear_cache()
        
        elapsed = time.time() - start_time
        print(f"Completed {num_shapes} shapes in {elapsed:.2f}s ({num_shapes/elapsed:.2f} shapes/sec)")
        
        return self.score_history[1:]  # Skip the initial score
    
    def save_image(self, output_path):
        """
        Save the current image.
        
        Args:
            output_path (str): Path to save the image
        """
        # Convert tensor to PIL image
        current_np = self.accelerator.to_numpy(self.current_image_tensor)
        # Convert from CHW to HWC format
        current_np = np.transpose(current_np, (1, 2, 0))
        current_np = np.clip(current_np * 255, 0, 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save image
        Image.fromarray(current_np).save(output_path)
        print(f"Saved image to {output_path}")
    
    def save_svg(self, output_path):
        """
        Save the current image as SVG.
        
        Args:
            output_path (str): Path to save the SVG
        """
        # Create SVG content
        bg_color = self._compute_background_color() * 255
        bg_color = [int(c) for c in bg_color]
        
        # Start SVG
        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">')
        
        # Add background
        lines.append(f'<rect width="{self.width}" height="{self.height}" fill="rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})" />')
        
        # Add shapes
        for i, shape in enumerate(self.shape_history):
            lines.append(shape.to_svg(self.color_history[i]))
        
        # End SVG
        lines.append('</svg>')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write SVG file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Saved SVG to {output_path}")
    
    def save_animation(self, output_path, frame_count=None):
        """
        Save an animated GIF showing the progression.
        
        Args:
            output_path (str): Path to save the animation
            frame_count (int): Number of frames to include (None for all shapes)
        """
        if frame_count is None:
            frame_count = len(self.shape_history)
        
        frame_count = min(frame_count, len(self.shape_history))
        
        # Determine which shapes to render for each frame
        if frame_count <= 1:
            indices = [len(self.shape_history)]
        else:
            indices = [int((i / (frame_count - 1)) * len(self.shape_history)) for i in range(frame_count)]
            indices[-1] = len(self.shape_history)  # Ensure we include the final shape
        
        # Render frames
        frames = []
        
        # Start with background
        bg_color = self._compute_background_color()
        current = self._create_background_image(bg_color)
        
        # First frame is just the background
        current_np = self.accelerator.to_numpy(current)
        current_np = np.transpose(current_np, (1, 2, 0))
        current_np = np.clip(current_np * 255, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(current_np))
        
        # Add shapes progressively
        for idx in indices:
            if idx == 0:
                continue
                
            # Reset to background
            current = self._create_background_image(bg_color)
            
            # Add shapes up to idx
            for i in range(idx):
                shape = self.shape_history[i]
                color = self.color_history[i]
                
                # Convert shape to mask
                mask = shape.to_tensor()
                
                # Convert RGB components to tensor
                rgb_tensor = torch.tensor([color['r'], color['g'], color['b']], device=self.accelerator.device) / 255.0
                alpha_value = color['a'] / 255.0
                
                # Expand mask and rgb tensor to match image dimensions for proper broadcasting
                mask_expanded = mask.unsqueeze(0).expand_as(current)
                rgb_expanded = rgb_tensor.view(-1, 1, 1).expand_as(current)
                
                # Apply alpha blending with proper broadcasting
                current = current * (1 - mask_expanded * alpha_value) + rgb_expanded * mask_expanded * alpha_value
            
            # Convert to PIL image
            current_np = self.accelerator.to_numpy(current)
            current_np = np.transpose(current_np, (1, 2, 0))
            current_np = np.clip(current_np * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(current_np))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save animation
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,
            loop=0
        )
        
        print(f"Saved animation to {output_path}") 