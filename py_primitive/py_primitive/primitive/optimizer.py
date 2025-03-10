"""
Particle Swarm Optimization for finding optimal shapes.
"""
import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Callable
import time

from py_primitive.primitive.shapes import Shape, create_random_shape

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for finding optimal shapes.
    
    This class implements a highly parallelized PSO algorithm optimized for GPU.
    It evaluates large batches of candidate shapes simultaneously to maximize GPU utilization.
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
        
        # PSO parameters
        self.swarm_size = config.get("swarm_size", 100)  # Larger swarm for better parallelism
        self.iterations = config.get("iterations", 10)
        self.cognitive_weight = config.get("cognitive_weight", 1.5)
        self.social_weight = config.get("social_weight", 1.5)
        self.inertia_weight = config.get("inertia_weight", 0.7)
        self.batch_size = config.get("batch_size", 128)  # Larger batch size for GPU
        
        # Keep track of velocity and position for each particle
        self.velocities = {}
        
    def _init_swarm(self, shape_type) -> List[Shape]:
        """
        Initialize a swarm of random shapes.
        
        Args:
            shape_type (int): Type of shape to create
            
        Returns:
            List[Shape]: Swarm of shapes
        """
        swarm = []
        # Create a large batch of shapes at once
        for _ in range(self.swarm_size):
            swarm.append(create_random_shape(shape_type, self.width, self.height, self.gpu))
        return swarm
    
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
        # Process in batches to avoid OOM errors
        all_scores = []
        all_colors = []
        
        for i in range(0, len(shapes), self.batch_size):
            batch = shapes[i:i+self.batch_size]
            batch_size = len(batch)
            
            # Convert shapes to tensor masks in parallel
            masks = torch.zeros((batch_size, 1, self.height, self.width), device=self.gpu.device)
            for j, shape in enumerate(batch):
                mask = shape.to_tensor()
                masks[j, 0] = mask
            
            # Compute optimal colors for all shapes in the batch simultaneously
            batch_colors = self._compute_batch_colors(target_image, current_image, masks, alpha)
            
            # Create candidate images by applying each shape with its optimal color
            candidates = self._apply_shapes_batch(current_image, masks, batch_colors, alpha)
            
            # Compute differences in parallel
            expanded_target = target_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
            differences = torch.mean((candidates - expanded_target) ** 2, dim=(1, 2, 3))
            
            # Add to results
            all_scores.extend(self.gpu.to_numpy(differences).tolist())
            all_colors.extend(batch_colors)
            
            # Clear GPU cache
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        return all_scores, all_colors
    
    def _compute_batch_colors(self, 
                             target: torch.Tensor, 
                             current: torch.Tensor, 
                             masks: torch.Tensor, 
                             alpha: int) -> List[Dict[str, int]]:
        """
        Compute optimal colors for multiple shapes in parallel.
        
        Args:
            target: Target image tensor
            current: Current image tensor
            masks: Batch of shape masks [batch_size, 1, height, width]
            alpha: Alpha value
            
        Returns:
            List[Dict[str, int]]: List of color dictionaries
        """
        batch_size = masks.shape[0]
        results = []
        alpha_value = alpha / 255.0
        
        # Calculate impact of each shape mask on the image
        # This is done for the entire batch at once
        masks_flat = masks.view(batch_size, 1, -1)  # [batch, 1, height*width]
        target_flat = target.view(3, -1).unsqueeze(0)  # [1, 3, height*width]
        current_flat = current.view(3, -1).unsqueeze(0)  # [1, 3, height*width]
        
        # Compute the weighted sum for each shape (color calculation)
        # This vectorizes the color computation for all shapes at once
        for i in range(batch_size):
            mask = masks[i, 0]
            if torch.sum(mask) < 1:
                # Empty shape, use default color
                results.append({'r': 0, 'g': 0, 'b': 0, 'a': alpha})
                continue
                
            # Extract active pixels for this mask
            mask_bool = mask.bool()
            target_pixels = target[:, mask_bool]
            current_pixels = current[:, mask_bool]
            
            # Compute optimal color using vectorized operations
            if alpha == 0:
                # Algorithm chooses alpha - simplified version for speed
                best_alpha = 128
                alpha_normalized = best_alpha / 255.0
                
                # Calculate the color that minimizes error
                numerator = torch.sum(target_pixels - current_pixels * (1 - alpha_normalized), dim=1)
                denominator = torch.sum(alpha_normalized * torch.ones_like(target_pixels[0]))
                color = numerator / (denominator + 1e-8)
                color = torch.clamp(color, 0, 1)
                
                alpha_value = best_alpha
            else:
                alpha_value = alpha
                alpha_normalized = alpha / 255.0
                
                # Calculate the color that minimizes error
                numerator = torch.sum(target_pixels - current_pixels * (1 - alpha_normalized), dim=1)
                denominator = torch.sum(alpha_normalized * torch.ones_like(target_pixels[0]))
                color = numerator / (denominator + 1e-8)
                color = torch.clamp(color, 0, 1)
            
            # Convert to 8-bit RGB values
            r, g, b = [int(c * 255) for c in self.gpu.to_numpy(color)]
            results.append({'r': r, 'g': g, 'b': b, 'a': alpha_value})
        
        return results
    
    def _apply_shapes_batch(self, 
                           current_image: torch.Tensor, 
                           masks: torch.Tensor, 
                           colors: List[Dict[str, int]], 
                           alpha: int) -> torch.Tensor:
        """
        Apply a batch of shapes to the current image in parallel.
        
        Args:
            current_image: Current image tensor
            masks: Batch of shape masks [batch_size, 1, height, width]
            colors: List of color dictionaries
            alpha: Alpha value
            
        Returns:
            torch.Tensor: Batch of candidate images [batch_size, channels, height, width]
        """
        batch_size = masks.shape[0]
        channels, height, width = current_image.shape
        
        # Create batch of RGB tensors from colors
        rgb_batch = torch.zeros((batch_size, 3), device=self.gpu.device)
        alpha_values = torch.zeros(batch_size, device=self.gpu.device)
        
        for i, color in enumerate(colors):
            rgb_batch[i, 0] = color['r'] / 255.0
            rgb_batch[i, 1] = color['g'] / 255.0
            rgb_batch[i, 2] = color['b'] / 255.0
            alpha_values[i] = color['a'] / 255.0
        
        # Expand current image for the batch
        current_expanded = current_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Expand masks for broadcasting with RGB channels
        masks_expanded = masks.expand(-1, 3, -1, -1)
        
        # Expand RGB values for broadcasting with image dimensions
        rgb_expanded = rgb_batch.view(batch_size, 3, 1, 1).expand(-1, -1, height, width)
        
        # Expand alpha values for broadcasting
        alpha_expanded = alpha_values.view(batch_size, 1, 1, 1).expand(-1, 3, height, width)
        
        # Apply alpha blending in a single vectorized operation
        result = current_expanded * (1 - masks_expanded * alpha_expanded) + rgb_expanded * masks_expanded * alpha_expanded
        
        return result
    
    def _update_velocities_and_positions(self, 
                                        particles: List[Shape], 
                                        particle_scores: List[float],
                                        best_global_particle: Shape) -> List[Shape]:
        """
        Update velocities and positions of particles in the swarm.
        
        Args:
            particles: List of shapes (particles)
            particle_scores: Scores of each particle
            best_global_particle: Best particle found so far
            
        Returns:
            List[Shape]: Updated particles
        """
        # Find personal best for each particle
        best_score_idx = np.argmin(particle_scores)
        best_global_score = particle_scores[best_score_idx]
        
        # If no velocities exist yet, initialize them
        if len(self.velocities) == 0:
            for i in range(len(particles)):
                self.velocities[i] = []
                
        # Update each particle's velocity and position
        new_particles = []
        for i, particle in enumerate(particles):
            # Initialize velocity if this is the first iteration
            if i not in self.velocities or not self.velocities[i]:
                # Initial velocity is just a small random mutation
                self.velocities[i] = [particle.mutate(rate=0.1)]
            
            # Current velocity
            velocity_shape = self.velocities[i][-1]
            
            # Randomly sample cognitive and social weights
            cognitive_random = random.uniform(0, 1)
            social_random = random.uniform(0, 1)
            
            # Create a new velocity by combining:
            # 1. Inertia component (previous velocity)
            # 2. Cognitive component (personal best)
            # 3. Social component (global best)
            
            # For the cognitive component, we use the particle itself (as we don't store personal bests)
            cognitive_component = particle.crossover(velocity_shape)
            
            # For the social component, we use the global best
            social_component = best_global_particle.crossover(velocity_shape)
            
            # Combine the components to get a new velocity
            # We simulate this by creating a shape that's influenced by all components
            inertia_mutation = velocity_shape.mutate(rate=self.inertia_weight)
            cognitive_mutation = cognitive_component.mutate(rate=self.cognitive_weight * cognitive_random)
            social_mutation = social_component.mutate(rate=self.social_weight * social_random)
            
            # New velocity is a combination of these components (via crossover)
            new_velocity = inertia_mutation.crossover(cognitive_mutation).crossover(social_mutation)
            
            # Update particle position (create a new particle with the new velocity)
            new_particle = particle.crossover(new_velocity)
            
            # Store the new velocity for the next iteration
            self.velocities[i].append(new_velocity)
            # Keep only the most recent velocity to save memory
            if len(self.velocities[i]) > 1:
                self.velocities[i] = self.velocities[i][-1:]
                
            new_particles.append(new_particle)
        
        return new_particles
    
    def find_best_shape(self, 
                       shape_type: int, 
                       target_image: torch.Tensor, 
                       current_image: torch.Tensor, 
                       alpha: int) -> Tuple[Shape, Dict[str, Any], float]:
        """
        Find the best shape to add to the current image using PSO.
        
        Args:
            shape_type (int): Type of shape to find
            target_image: Target image tensor
            current_image: Current image tensor
            alpha (int): Alpha value for the shape
            
        Returns:
            Tuple[Shape, Dict[str, Any], float]: Best shape, color, and score
        """
        start_time = time.time()
        
        # Initialize swarm
        particles = self._init_swarm(shape_type)
        
        # Evaluate initial swarm
        scores, colors = self._evaluate_batch(particles, target_image, current_image, alpha)
        
        # Find best particle
        best_idx = np.argmin(scores)
        best_particle = particles[best_idx]
        best_color = colors[best_idx]
        best_score = scores[best_idx]
        
        # Track if score improves
        last_best_score = best_score
        no_improvement_count = 0
        max_no_improvement = 2  # Stop after 2 iterations without improvement
        
        # Main PSO loop
        for iteration in range(self.iterations):
            # Update velocities and positions
            particles = self._update_velocities_and_positions(particles, scores, best_particle)
            
            # Evaluate updated particles
            scores, colors = self._evaluate_batch(particles, target_image, current_image, alpha)
            
            # Update best particle if found better
            new_best_idx = np.argmin(scores)
            new_best_score = scores[new_best_idx]
            
            if new_best_score < best_score:
                best_idx = new_best_idx
                best_particle = particles[best_idx]
                best_color = colors[best_idx]
                best_score = new_best_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Early stopping check
            if no_improvement_count >= max_no_improvement:
                print(f"Early stopping at iteration {iteration+1}/{self.iterations} - no improvement for {max_no_improvement} iterations")
                break
        
        elapsed = time.time() - start_time
        print(f"Particle Swarm Optimization completed in {elapsed:.2f}s, best score: {best_score:.6f}")
        
        return best_particle, best_color, best_score 