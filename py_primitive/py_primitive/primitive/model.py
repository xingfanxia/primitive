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
from svgpathtools import parse_path, Line
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
        # Check for GPU availability (either CUDA or MPS)
        gpu_available = (hasattr(torch, 'cuda') and torch.cuda.is_available()) or \
                      (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        
        self.use_gpu = self.config.get("use_gpu", True) and gpu_available
        
        if self.use_gpu:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using GPU acceleration with MPS (Metal Performance Shaders)")
                self.accelerator = GPUAccelerator()
            elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                print("Using GPU acceleration with CUDA")
                self.accelerator = GPUAccelerator()
            else:
                print("GPU requested but not available, falling back to CPU")
                self.accelerator = CPUAccelerator()
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
        
        # Store original image dimensions before resizing
        self.original_width = None
        self.original_height = None
        
        # Get original dimensions before resizing
        with Image.open(target_image_path) as img:
            self.original_width, self.original_height = img.size

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
        
        # Create PIL image
        img = Image.fromarray(current_np)
        
        # Determine output size - use original dimensions by default
        output_size = self.config.get("output_size", None)
        if output_size and output_size > 0:
            # User specifically requested an output size
            if isinstance(output_size, int):
                # Single value - maintain aspect ratio
                w, h = img.size
                if w > h:
                    new_w = output_size
                    new_h = int(h * output_size / w)
                else:
                    new_h = output_size
                    new_w = int(w * output_size / h)
            elif isinstance(output_size, tuple) and len(output_size) == 2:
                # Exact dimensions specified
                new_w, new_h = output_size
            else:
                # Invalid format - use original dimensions
                new_w, new_h = self.original_width, self.original_height
        else:
            # No output size specified - use original dimensions
            new_w, new_h = self.original_width, self.original_height
        
        # Resize if needed
        if (new_w, new_h) != img.size:
            print(f"Resizing output image to {new_w}x{new_h} pixels...")
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save image
        img.save(output_path)
        print(f"Saved image to {output_path}")
    
    def save_svg(self, output_path):
        """
        Save the current image as SVG.
        
        Args:
            output_path (str): Path to save the SVG
        """
        # Get dimensions with respect to potential output scaling
        output_size = self.config.get("output_size", None)
        
        # Default to original dimensions
        w, h = self.original_width, self.original_height
        
        # If output size is specified, use that instead
        if output_size and output_size > 0:
            if isinstance(output_size, int):
                # Single value - maintain aspect ratio
                if self.original_width > self.original_height:
                    w = output_size
                    h = int(self.original_height * output_size / self.original_width)
                else:
                    h = output_size
                    w = int(self.original_width * output_size / self.original_height)
            elif isinstance(output_size, tuple) and len(output_size) == 2:
                w, h = output_size
        
        # Create SVG content
        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
        
        # Add background rectangle with the background color
        bg_color = self.bg_color
        r, g, b = [int(c * 255) for c in bg_color]
        lines.append(f'  <rect width="{w}" height="{h}" fill="rgb({r}, {g}, {b})"/>')
        
        # Add shapes - scale them to match the output dimensions
        scale_x = w / self.width
        scale_y = h / self.height
        
        for i, shape in enumerate(self.shape_history):
            # Get SVG for the shape and scale if needed
            svg_shape = shape.to_svg(self.color_history[i])
            if scale_x != 1 or scale_y != 1:
                # Add transform attribute to scale the shape
                svg_shape = svg_shape.replace('>', f' transform="scale({scale_x}, {scale_y})">', 1)
            lines.append(svg_shape)
        
        # End SVG
        lines.append('</svg>')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write SVG to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Saved SVG to {output_path}")
    
    def save_animation(self, output_path, frame_count=None, fps=5, max_size=512):
        """
        Save an animated GIF showing the progression.
        
        Args:
            output_path (str): Path to save the animation
            frame_count (int): Number of frames to include (None for all shapes)
            fps (int): Frames per second for the animation (default: 5)
            max_size (int): Maximum size for the animation (default: 512px)
        """
        # Calculate duration in milliseconds from fps
        duration = int(1000 / fps)
        
        # Determine which shapes to render for each frame
        total_shapes = len(self.shape_history)
        
        # Always start with background
        indices = [0]
        
        # First 10 shapes: include every shape
        end_idx = min(10, total_shapes)
        indices.extend(range(1, end_idx + 1))
        
        if total_shapes > 10:
            # Shapes 11-30: every 2nd shape
            start_idx = 11
            end_idx = min(30, total_shapes)
            indices.extend(range(start_idx, end_idx + 1, 2))
            
            if total_shapes > 30:
                # Shapes 31-50: every 3rd shape
                start_idx = 31
                end_idx = min(50, total_shapes)
                indices.extend(range(start_idx, end_idx + 1, 3))
                
                if total_shapes > 50:
                    # Remaining shapes: every 5th shape
                    start_idx = 51
                    indices.extend(range(start_idx, total_shapes + 1, 5))
        
        # Always ensure we include the final shape
        if total_shapes not in indices:
            indices.append(total_shapes)
        
        # Ensure indices are unique and sorted
        indices = sorted(list(set(indices)))
        
        print(f"Creating animation with {len(indices)} frames...")
        
        # Render frames
        frames = []
        
        # Start with background
        bg_color = self._compute_background_color()
        current = self._create_background_image(bg_color)
        
        # Convert background to PIL image
        current_np = self.accelerator.to_numpy(current)
        current_np = np.transpose(current_np, (1, 2, 0))
        current_np = np.clip(current_np * 255, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(current_np))
        
        # Track the last rendered index to avoid duplicating work
        last_rendered_idx = 0
        
        # Add shapes progressively
        for idx in sorted(indices)[1:]:  # Skip 0 as we already rendered the background
            if idx <= last_rendered_idx:
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
            
            last_rendered_idx = idx
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine output size - use original dimensions by default
        output_size = self.config.get("output_size", None)
        
        # Get input resize value to use as a reference for max size
        input_resize = self.config.get("input_resize", 256)
        
        # Use the smaller of max_size or input_resize as our limit
        size_limit = min(max_size, input_resize)
        
        if output_size and output_size > 0:
            # User specifically requested an output size
            if isinstance(output_size, int):
                # Single value - maintain aspect ratio
                w, h = frames[0].size
                if w > h:
                    new_w = min(output_size, size_limit)
                    new_h = int(h * new_w / w)
                else:
                    new_h = min(output_size, size_limit)
                    new_w = int(w * new_h / h)
            elif isinstance(output_size, tuple) and len(output_size) == 2:
                # Exact dimensions specified
                orig_w, orig_h = output_size
                # Scale down if needed
                if max(orig_w, orig_h) > size_limit:
                    scale = size_limit / max(orig_w, orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                else:
                    new_w, new_h = orig_w, orig_h
            else:
                # Invalid format - use original dimensions but limit size
                w, h = self.original_width, self.original_height
                if max(w, h) > size_limit:
                    scale = size_limit / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                else:
                    new_w, new_h = w, h
        else:
            # No output size specified - use original dimensions but limit size
            w, h = self.original_width, self.original_height
            if max(w, h) > size_limit:
                scale = size_limit / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w, new_h = w, h
        
        # Resize frames if needed
        if (new_w, new_h) != frames[0].size:
            print(f"Resizing animation frames to {new_w}x{new_h} pixels...")
            resized_frames = []
            for frame in frames:
                resized_frames.append(frame.resize((new_w, new_h), Image.LANCZOS))
            frames = resized_frames
        
        # Save animation
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,  # Enable optimization to reduce file size
            duration=duration,
            loop=0
        )
        
        print(f"Saved animation to {output_path} (FPS: {fps}, Size: {new_w}x{new_h} pixels)") 