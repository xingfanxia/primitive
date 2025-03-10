"""
GPU Acceleration utilities using PyTorch and MPS (Metal Performance Shaders).
"""
import torch
import numpy as np

class GPUAccelerator:
    """
    Handles GPU acceleration using PyTorch with MPS backend on M-series Macs.
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize the GPU accelerator.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and self._is_mps_available()
        self.device = self._get_device()
        
    def _is_mps_available(self):
        """Check if MPS (Metal Performance Shaders) is available."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    def _get_device(self):
        """Get the appropriate device (MPS or CPU)."""
        if self.use_gpu:
            return torch.device("mps")
        return torch.device("cpu")
    
    def to_tensor(self, data, dtype=torch.float32):
        """
        Convert numpy array or list to tensor and move to the appropriate device.
        
        Args:
            data: Numpy array or list to convert
            dtype: Tensor data type
            
        Returns:
            torch.Tensor: Tensor on the appropriate device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype)
        return torch.tensor(data, dtype=dtype, device=self.device)
    
    def to_numpy(self, tensor):
        """
        Convert a tensor to numpy array.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            numpy.ndarray: Numpy array
        """
        if tensor.device.type == 'mps':
            return tensor.detach().cpu().numpy()
        return tensor.detach().numpy()
    
    def batch_process(self, func, inputs, batch_size=32):
        """
        Process inputs in batches to avoid memory issues.
        
        Args:
            func: Function to apply to each batch
            inputs: List of inputs
            batch_size: Size of each batch
            
        Returns:
            list: Processed outputs
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_results = func(batch)
            results.extend(batch_results)
        return results
    
    def compute_image_difference(self, target, current, affected_area=None):
        """
        Compute MSE difference between target and current images.
        
        Args:
            target: Target image tensor
            current: Current image tensor
            affected_area: Optional mask of affected pixels
            
        Returns:
            float: Mean squared error
        """
        if affected_area is not None:
            # Compute difference only for affected pixels
            mask = self.to_tensor(affected_area)
            diff = (target - current) ** 2
            masked_diff = diff * mask
            return torch.sum(masked_diff) / (torch.sum(mask) + 1e-8)
        else:
            # Compute difference for the entire image
            return torch.mean((target - current) ** 2).item()
    
    def compute_batch_differences(self, target, current, candidates):
        """
        Compute differences for multiple candidate images in parallel.
        
        Args:
            target: Target image tensor
            current: Current image tensor
            candidates: Batch of candidate shapes to evaluate
            
        Returns:
            torch.Tensor: Tensor of difference scores
        """
        # Stack all candidates into a single batch tensor
        batch_size = len(candidates)
        
        # Convert candidates to tensors representing shape masks
        # (Implementation depends on shape representation)
        
        # Example assuming candidates are already converted to image tensors:
        candidate_tensors = torch.stack(candidates)
        
        # Compute differences in parallel
        expanded_target = target.expand(batch_size, -1, -1, -1)
        expanded_current = current.expand(batch_size, -1, -1, -1)
        
        # Mean squared error calculation
        differences = torch.mean((expanded_target - candidate_tensors) ** 2, dim=(1, 2, 3))
        
        return differences 