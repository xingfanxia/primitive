"""
GPU-optimized randomized search for finding optimal shapes.
"""
import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import time

from py_primitive.primitive.shapes import Shape, create_random_shape

class RandomizedShapeOptimizer:
    """
    Fast GPU-optimized randomized search for finding optimal shapes.
    
    This implements a parallelized random search with local refinement.
    It evaluates a large batch of random candidates at once, then refines
    the best few candidates with small perturbations.
    """
    
    def __init__(self, config, gpu_accelerator, image_width, image_height):
        """
        Initialize the optimizer.
        
        Args:
            config (dict): Configuration parameters
            gpu_accelerator: GPU accelerator for tensor operations
            image_width (int): Width of the image
            image_height (int): Height of the image
        """
        self.config = config
        self.gpu = gpu_accelerator
        self.width = image_width
        self.height = image_height
        
        # Optimization parameters
        self.candidates = config.get("candidates", 75)  # Number of initial random candidates
        self.refinements = config.get("refinements", 5)  # Number of refinement steps
        self.top_candidates = config.get("top_candidates", 3)  # Number of top candidates to refine
        self.batch_size = config.get("batch_size", 75)  # Process in batches
        
    def _generate_candidates(self, shape_type, count) -> List[Shape]:
        """
        Generate random shape candidates.
        
        Args:
            shape_type (int): Type of shape to create
            count (int): Number of candidates to generate
            
        Returns:
            List[Shape]: List of random shape candidates
        """
        return [create_random_shape(shape_type, self.width, self.height, self.gpu) 
                for _ in range(count)]
    
    def _evaluate_batch(self, 
                      shapes: List[Shape], 
                      target_image: torch.Tensor, 
                      current_image: torch.Tensor, 
                      alpha: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate a batch of shapes in parallel on the GPU.
        
        Args:
            shapes (List[Shape]): Shapes to evaluate
            target_image: Target image tensor
            current_image: Current image tensor
            alpha (int): Alpha value for the shapes
            
        Returns:
            Tuple[List[float], List[Dict[str, Any]]]: Scores and color information
        """
        batch_size = len(shapes)
        
        # Fast path for small batches - process all at once
        if batch_size <= self.batch_size:
            return self._evaluate_shapes(shapes, target_image, current_image, alpha)
        
        # Process in batches for larger sets
        all_scores = []
        all_colors = []
        
        for i in range(0, batch_size, self.batch_size):
            batch = shapes[i:i+self.batch_size]
            scores, colors = self._evaluate_shapes(batch, target_image, current_image, alpha)
            all_scores.extend(scores)
            all_colors.extend(colors)
        
        return all_scores, all_colors
    
    def _evaluate_shapes(self, 
                        shapes: List[Shape], 
                        target_image: torch.Tensor, 
                        current_image: torch.Tensor, 
                        alpha: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Core evaluation function for a small batch of shapes.
        
        Args:
            shapes: List of shapes to evaluate
            target_image: Target image tensor
            current_image: Current image tensor
            alpha: Alpha value
            
        Returns:
            Tuple[List[float], List[Dict[str, Any]]]: Scores and color information
        """
        batch_size = len(shapes)
        
        # Fast path for empty batch
        if batch_size == 0:
            return [], []
        
        # Pre-allocate tensor for all masks at once
        masks = torch.zeros((batch_size, self.height, self.width), 
                           device=self.gpu.device, dtype=torch.float32)
        
        # Generate all masks in parallel
        for i, shape in enumerate(shapes):
            masks[i] = shape.to_tensor()
        
        # Compute best colors for all shapes at once
        alpha_value = alpha / 255.0
        colors = []
        
        for i in range(batch_size):
            mask = masks[i]
            
            # Skip empty shapes
            if torch.sum(mask) < 1:
                colors.append({'r': 128, 'g': 128, 'b': 128, 'a': alpha})
                continue
            
            # Extract pixels inside the shape
            indices = mask > 0
            target_pixels = target_image[:, indices]
            current_pixels = current_image[:, indices]
            
            # For small shapes, use a simpler color computation
            if target_pixels.shape[1] < 10:
                rgb = torch.mean(target_image, dim=(1, 2))
                r, g, b = [int(c * 255) for c in self.gpu.to_numpy(rgb)]
                colors.append({'r': r, 'g': g, 'b': b, 'a': alpha})
                continue
            
            # Calculate optimal color
            color = self._compute_optimal_color(target_pixels, current_pixels, alpha_value)
            r, g, b = [int(c * 255) for c in self.gpu.to_numpy(color)]
            colors.append({'r': r, 'g': g, 'b': b, 'a': alpha})
        
        # Compute scores in a single batched operation
        scores = self._compute_batch_scores(masks, colors, target_image, current_image)
        
        return scores, colors
    
    def _compute_optimal_color(self, target_pixels, current_pixels, alpha_value):
        """
        Compute the optimal color for a set of pixels.
        
        Args:
            target_pixels: Target image pixels [3, N]
            current_pixels: Current image pixels [3, N]
            alpha_value: Alpha value (0-1)
            
        Returns:
            torch.Tensor: Optimal RGB color [3]
        """
        # Simple weighted average for optimal color
        numerator = torch.sum(target_pixels - current_pixels * (1 - alpha_value), dim=1)
        denominator = alpha_value * target_pixels.shape[1]
        color = numerator / (denominator + 1e-6)
        return torch.clamp(color, 0, 1)
    
    def _compute_batch_scores(self, masks, colors, target_image, current_image):
        """
        Compute scores for multiple shapes in parallel.
        
        Args:
            masks: Shape masks [batch_size, height, width]
            colors: List of color dictionaries
            target_image: Target image tensor
            current_image: Current image tensor
            
        Returns:
            List[float]: Scores for each shape
        """
        batch_size = masks.shape[0]
        
        # Pre-allocate tensors for candidates
        candidates = torch.zeros((batch_size, *current_image.shape), 
                                device=self.gpu.device, dtype=torch.float32)
        
        # Apply each mask with its color
        for i in range(batch_size):
            mask = masks[i]
            color = colors[i]
            
            # Convert color to tensor
            rgb = torch.tensor([color['r'], color['g'], color['b']], 
                              device=self.gpu.device, dtype=torch.float32) / 255.0
            alpha_value = color['a'] / 255.0
            
            # Apply mask with color to current image
            for c in range(3):  # RGB channels
                candidates[i, c] = current_image[c] * (1 - mask * alpha_value) + rgb[c] * mask * alpha_value
        
        # Compute MSE difference in a single batched operation
        target_expanded = target_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        mse = torch.mean((candidates - target_expanded) ** 2, dim=(1, 2, 3))
        
        # Convert to list
        return mse.cpu().numpy().tolist()
    
    def find_best_shape(self, 
                       shape_type: int, 
                       target_image: torch.Tensor, 
                       current_image: torch.Tensor, 
                       alpha: int) -> Tuple[Shape, Dict[str, Any], float]:
        """
        Find the best shape using a fast GPU-accelerated random search.
        
        Args:
            shape_type (int): Type of shape to find
            target_image: Target image tensor
            current_image: Current image tensor
            alpha (int): Alpha value for the shape
            
        Returns:
            Tuple[Shape, Dict[str, Any], float]: Best shape, color, and score
        """
        start_time = time.time()
        
        # Phase 1: Generate and evaluate random candidates
        candidates = self._generate_candidates(shape_type, self.candidates)
        scores, colors = self._evaluate_batch(candidates, target_image, current_image, alpha)
        
        # Find the top K candidates
        indices = np.argsort(scores)[:self.top_candidates]
        best_candidates = [candidates[i] for i in indices]
        best_scores = [scores[i] for i in indices]
        best_colors = [colors[i] for i in indices]
        
        best_idx = 0  # Best is the first one after sorting
        best_shape = best_candidates[best_idx]
        best_color = best_colors[best_idx]
        best_score = best_scores[best_idx]
        
        # Phase 2: Refine the top candidates with small mutations
        for refinement in range(self.refinements):
            # Generate mutations for each top candidate
            refined_candidates = []
            
            for i, candidate in enumerate(best_candidates):
                # Generate 10 mutations for each top candidate
                mutations = [candidate.mutate(rate=0.1 / (refinement + 1)) for _ in range(10)]
                refined_candidates.extend(mutations)
            
            # Evaluate refined candidates
            refined_scores, refined_colors = self._evaluate_batch(
                refined_candidates, target_image, current_image, alpha
            )
            
            # Check if we found a better score
            best_refined_idx = np.argmin(refined_scores)
            if refined_scores[best_refined_idx] < best_score:
                best_shape = refined_candidates[best_refined_idx]
                best_color = refined_colors[best_refined_idx]
                best_score = refined_scores[best_refined_idx]
                
                # Sort and update best candidates
                indices = np.argsort(refined_scores)[:self.top_candidates]
                best_candidates = [refined_candidates[i] for i in indices]
                best_scores = [refined_scores[i] for i in indices]
                best_colors = [refined_colors[i] for i in indices]
            else:
                # If no improvement, reduce search space and try again with smaller mutations
                pass
        
        elapsed = time.time() - start_time
        print(f"Optimized search completed in {elapsed:.2f}s, best score: {best_score:.6f}")
        
        return best_shape, best_color, best_score 