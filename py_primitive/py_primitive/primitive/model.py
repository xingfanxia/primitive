"""
Main model for generating primitive images.
"""
import torch
import numpy as np
from PIL import Image
import time
import os
from typing import List, Dict, Any, Tuple

from py_primitive.primitive.gpu import GPUAccelerator
from py_primitive.primitive.optimizer import ParticleSwarmOptimizer
from py_primitive.primitive.shapes import Shape
from py_primitive.config.config import get_config, SHAPE_TYPES

class PrimitiveModel:
    """
    Main model for generating primitive images using GPU acceleration.
    """
    
    def __init__(self, target_image_path, config=None):
        """
        Initialize the model.
        
        Args:
            target_image_path (str): Path to the target image
            config (dict): Configuration parameters (optional)
        """
        # Load configuration
        self.config = get_config(config)
        
        # Initialize GPU accelerator
        self.gpu = GPUAccelerator(use_gpu=self.config["use_gpu"])
        
        # Load and process target image
        self.target_pil = self._load_and_resize_image(target_image_path)
        self.width, self.height = self.target_pil.size
        
        # Convert image to tensor
        target_np = np.array(self.target_pil) / 255.0
        # Convert from HWC to CHW format
        target_np = np.transpose(target_np, (2, 0, 1))
        self.target = self.gpu.to_tensor(target_np)
        
        # Create current image with background color
        bg_color = self._compute_background_color()
        self.current = self._create_background_image(bg_color)
        
        # Create optimizer
        self.optimizer = ParticleSwarmOptimizer(
            self.config, 
            self.gpu, 
            self.width, 
            self.height
        )
        
        # Initialize state
        self.shapes = []
        self.colors = []
        self.scores = [self._compute_score()]
        
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
        
        return img
        
    def _compute_background_color(self):
        """
        Compute the average color of the target image.
        
        Returns:
            np.ndarray: Background color as RGB values in [0,1]
        """
        # Compute average color across all pixels
        avg_color = torch.mean(self.target, dim=(1, 2))
        return self.gpu.to_numpy(avg_color)
    
    def _create_background_image(self, bg_color):
        """
        Create a background image filled with the specified color.
        
        Args:
            bg_color: Background color as RGB values in [0,1]
            
        Returns:
            torch.Tensor: Background image tensor
        """
        # Convert the background color to a tensor with the correct shape for broadcasting
        bg_tensor = self.gpu.to_tensor(bg_color).view(-1, 1, 1)
        
        # Create a tensor of the same shape as target filled with the background color
        return bg_tensor.expand_as(self.target)
    
    def _compute_score(self):
        """
        Compute the current score (difference between target and current).
        
        Returns:
            float: Score as mean squared error
        """
        return self.gpu.compute_image_difference(self.target, self.current)
    
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
            self.target,
            self.current,
            alpha
        )
        
        # Add the shape to the current image
        self._add_shape(shape, color)
        
        elapsed = time.time() - start_time
        shapes_per_sec = 1.0 / elapsed
        
        print(f"Added {SHAPE_TYPES[shape_type]}, score={score:.6f}, time={elapsed:.2f}s, shapes/sec={shapes_per_sec:.2f}")
        
        # Clear GPU cache periodically
        if hasattr(self.gpu, '_clear_cache'):
            self.gpu._clear_cache()
        
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
        rgb_tensor = torch.tensor([color['r'], color['g'], color['b']], device=self.gpu.device) / 255.0
        alpha_value = color['a'] / 255.0
        
        # Expand mask and rgb tensor to match image dimensions for proper broadcasting
        mask_expanded = mask.unsqueeze(0).expand_as(self.current)
        rgb_expanded = rgb_tensor.view(-1, 1, 1).expand_as(self.current)
        
        # Apply alpha blending with proper broadcasting
        self.current = self.current * (1 - mask_expanded * alpha_value) + rgb_expanded * mask_expanded * alpha_value
        
        # Update state
        self.shapes.append(shape)
        self.colors.append(color)
        self.scores.append(self._compute_score())
    
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
        print(f"Using acceleration: {self.gpu.device}")
        
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
            if hasattr(self.gpu, '_clear_cache'):
                self.gpu._clear_cache()
        
        elapsed = time.time() - start_time
        print(f"Completed {num_shapes} shapes in {elapsed:.2f}s ({num_shapes/elapsed:.2f} shapes/sec)")
        
        return self.scores[1:]  # Skip the initial score
    
    def save_image(self, output_path):
        """
        Save the current image.
        
        Args:
            output_path (str): Path to save the image
        """
        # Convert tensor to PIL image
        current_np = self.gpu.to_numpy(self.current)
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
        for i, shape in enumerate(self.shapes):
            lines.append(shape.to_svg(self.colors[i]))
        
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
            frame_count = len(self.shapes)
        
        frame_count = min(frame_count, len(self.shapes))
        
        # Determine which shapes to render for each frame
        if frame_count <= 1:
            indices = [len(self.shapes)]
        else:
            indices = [int((i / (frame_count - 1)) * len(self.shapes)) for i in range(frame_count)]
            indices[-1] = len(self.shapes)  # Ensure we include the final shape
        
        # Render frames
        frames = []
        
        # Start with background
        bg_color = self._compute_background_color()
        current = self._create_background_image(bg_color)
        
        # First frame is just the background
        current_np = self.gpu.to_numpy(current)
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
                shape = self.shapes[i]
                color = self.colors[i]
                
                # Convert shape to mask
                mask = shape.to_tensor()
                
                # Convert RGB components to tensor
                rgb_tensor = torch.tensor([color['r'], color['g'], color['b']], device=self.gpu.device) / 255.0
                alpha_value = color['a'] / 255.0
                
                # Expand mask and rgb tensor to match image dimensions for proper broadcasting
                mask_expanded = mask.unsqueeze(0).expand_as(current)
                rgb_expanded = rgb_tensor.view(-1, 1, 1).expand_as(current)
                
                # Apply alpha blending with proper broadcasting
                current = current * (1 - mask_expanded * alpha_value) + rgb_expanded * mask_expanded * alpha_value
            
            # Convert to PIL image
            current_np = self.gpu.to_numpy(current)
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