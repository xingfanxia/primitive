"""
Differential Evolution optimizer for finding optimal shapes.
"""
import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Callable
import time

from py_primitive.primitive.shapes import Shape, create_random_shape

class DifferentialEvolution:
    """
    Differential Evolution optimizer for finding optimal shapes.
    
    This class implements a parallel differential evolution algorithm optimized for GPU.
    It evaluates multiple candidate shapes simultaneously to find the best shape to add
    to the current image.
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
        
        # DE parameters
        self.population_size = config["population_size"]
        self.crossover_probability = config["crossover_probability"]
        self.mutation_factor = config["mutation_factor"]
        self.generations = config["generations"]
        
    def _init_population(self, shape_type) -> List[Shape]:
        """
        Initialize a population of random shapes.
        
        Args:
            shape_type (int): Type of shape to create
            
        Returns:
            List[Shape]: Population of shapes
        """
        return [create_random_shape(shape_type, self.width, self.height, self.gpu) 
                for _ in range(self.population_size)]
    
    def _evaluate_batch(self, 
                       population: List[Shape], 
                       target_image: torch.Tensor, 
                       current_image: torch.Tensor, 
                       alpha: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate a batch of shapes in parallel on the GPU.
        
        Args:
            population (List[Shape]): Shapes to evaluate
            target_image: Target image tensor
            current_image: Current image tensor
            alpha (int): Alpha value for the shapes
            
        Returns:
            Tuple[List[float], List[Dict[str, Any]]]: Scores and color information
        """
        # Convert shapes to tensor masks in parallel
        batch_size = len(population)
        masks = []
        for shape in population:
            mask = shape.to_tensor()
            # Add batch and channel dimensions
            mask = mask.unsqueeze(0).unsqueeze(0)
            masks.append(mask)
        
        # Stack masks into a batch
        stacked_masks = torch.cat(masks, dim=0)
        
        # Compute optimal colors for each shape
        colors = []
        for i in range(batch_size):
            color_dict = self._compute_optimal_color(target_image, current_image, stacked_masks[i][0], alpha)
            colors.append(color_dict)
        
        # Create candidate images by applying each shape with its optimal color
        candidates = []
        for i in range(batch_size):
            # Apply shape with computed color
            color = colors[i]
            rgb_tensor = torch.tensor([color['r'], color['g'], color['b']], device=self.gpu.device) / 255.0
            alpha_value = color['a'] / 255.0
            
            # Apply shape to current image
            mask = stacked_masks[i][0].unsqueeze(-1)  # Add channel dimension for broadcasting
            rgb_mask = mask * rgb_tensor
            
            # Apply alpha blending
            candidate = current_image * (1 - mask * alpha_value) + rgb_mask * alpha_value
            candidates.append(candidate)
        
        # Stack candidates into a batch for parallel evaluation
        stacked_candidates = torch.stack(candidates)
        
        # Compute differences in parallel
        expanded_target = target_image.expand(batch_size, -1, -1, -1)
        differences = torch.mean((stacked_candidates - expanded_target) ** 2, dim=(1, 2, 3))
        
        return self.gpu.to_numpy(differences).tolist(), colors
    
    def _compute_optimal_color(self, target, current, shape_mask, alpha) -> Dict[str, int]:
        """
        Compute the optimal color for a shape.
        
        Args:
            target: Target image tensor
            current: Current image tensor
            shape_mask: Binary mask tensor for the shape
            alpha (int): Alpha value
            
        Returns:
            Dict[str, int]: Optimal color with r, g, b, a components
        """
        # Only consider pixels inside the shape
        if torch.sum(shape_mask) < 1:
            # Shape is empty, use default color
            return {'r': 0, 'g': 0, 'b': 0, 'a': alpha}
        
        # Get the pixels that are inside the shape
        mask = shape_mask.bool()
        
        # Extract target and current pixels inside the shape
        target_pixels = target[:, mask]  # Shape: [3, num_pixels]
        current_pixels = current[:, mask]  # Shape: [3, num_pixels]
        
        # Calculate the weighted average color
        if alpha == 0:
            # Algorithm chooses alpha
            alpha_tensor = torch.linspace(8, 255, 32, device=self.gpu.device)
            best_alpha = 128
            best_score = float('inf')
            
            for a in alpha_tensor:
                a_normalized = a / 255.0
                # Calculate the color that minimizes error
                numerator = torch.sum(target_pixels - current_pixels * (1 - a_normalized), dim=1)
                denominator = torch.sum(a_normalized * torch.ones_like(target_pixels[0]))
                color = numerator / (denominator + 1e-8)
                
                # Clamp colors to valid range
                color = torch.clamp(color, 0, 1)
                
                # Calculate error with this color
                new_pixels = current_pixels * (1 - a_normalized) + color.unsqueeze(1) * a_normalized
                error = torch.sum((target_pixels - new_pixels) ** 2)
                
                if error < best_score:
                    best_score = error
                    best_alpha = a
                    best_color = color
            
            alpha_value = int(best_alpha.item())
            color = best_color
            
        else:
            # Use the provided alpha
            alpha_value = alpha
            alpha_normalized = alpha / 255.0
            
            # Calculate the color that minimizes error
            numerator = torch.sum(target_pixels - current_pixels * (1 - alpha_normalized), dim=1)
            denominator = torch.sum(alpha_normalized * torch.ones_like(target_pixels[0]))
            color = numerator / (denominator + 1e-8)
            
            # Clamp colors to valid range
            color = torch.clamp(color, 0, 1)
        
        # Convert to 8-bit RGB values
        r, g, b = [int(c * 255) for c in self.gpu.to_numpy(color)]
        
        return {'r': r, 'g': g, 'b': b, 'a': alpha_value}
    
    def _create_trial(self, target_idx, population) -> Shape:
        """
        Create a trial individual using differential evolution.
        
        Args:
            target_idx (int): Index of the target individual
            population (List[Shape]): Current population
            
        Returns:
            Shape: Trial individual
        """
        # Select three random individuals different from the target
        available_indices = [i for i in range(len(population)) if i != target_idx]
        a_idx, b_idx, c_idx = random.sample(available_indices, 3)
        
        a, b, c = population[a_idx], population[b_idx], population[c_idx]
        target = population[target_idx]
        
        # Create mutant by mutation
        # Instead of direct mutation, we're creating a new shape by crossing over
        # This is because shapes aren't directly mutable with vector operations
        
        # First create a mutated version of 'a' influenced by the difference between b and c
        mutant = a.mutate(rate=self.mutation_factor)
        
        # Crossover between target and mutant
        if random.random() < self.crossover_probability:
            trial = mutant.crossover(target)
        else:
            trial = target.mutate(rate=0.05)  # Small mutation if no crossover
        
        return trial
    
    def find_best_shape(self, 
                       shape_type: int, 
                       target_image: torch.Tensor, 
                       current_image: torch.Tensor, 
                       alpha: int) -> Tuple[Shape, Dict[str, Any], float]:
        """
        Find the best shape to add to the current image.
        
        Args:
            shape_type (int): Type of shape to find
            target_image: Target image tensor
            current_image: Current image tensor
            alpha (int): Alpha value for the shape
            
        Returns:
            Tuple[Shape, Dict[str, Any], float]: Best shape, color, and score
        """
        start_time = time.time()
        
        # Initialize population
        population = self._init_population(shape_type)
        
        # Evaluate initial population
        scores, colors = self._evaluate_batch(population, target_image, current_image, alpha)
        
        # Find best individual
        best_idx = np.argmin(scores)
        best_shape = population[best_idx]
        best_color = colors[best_idx]
        best_score = scores[best_idx]
        
        # Evolve the population
        for generation in range(self.generations):
            # Create and evaluate trials
            trials = []
            for i in range(self.population_size):
                trial = self._create_trial(i, population)
                trials.append(trial)
            
            trial_scores, trial_colors = self._evaluate_batch(trials, target_image, current_image, alpha)
            
            # Selection
            for i in range(self.population_size):
                if trial_scores[i] < scores[i]:
                    population[i] = trials[i]
                    scores[i] = trial_scores[i]
                    colors[i] = trial_colors[i]
                    
                    # Update best if needed
                    if trial_scores[i] < best_score:
                        best_idx = i
                        best_shape = trials[i]
                        best_color = trial_colors[i]
                        best_score = trial_scores[i]
        
        elapsed = time.time() - start_time
        print(f"Differential evolution completed in {elapsed:.2f}s, best score: {best_score:.6f}")
        
        return best_shape, best_color, best_score 