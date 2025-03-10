"""
Accelerator classes for GPU and CPU computation
"""
import torch
import numpy as np

class Accelerator:
    """Base accelerator class"""
    
    def __init__(self):
        self.device = None
    
    def to_tensor(self, array):
        """Convert numpy array to tensor on the device"""
        raise NotImplementedError
    
    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        raise NotImplementedError
    
    def compute_image_difference(self, target, current):
        """Compute the mean squared error between two images"""
        raise NotImplementedError
    
    def _clear_cache(self):
        """Clear device cache if supported"""
        pass

class GPUAccelerator(Accelerator):
    """GPU accelerator for tensor operations"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_tensor(self, array):
        """Convert numpy array to tensor on GPU"""
        if isinstance(array, np.ndarray):
            # Handle different dimensions properly
            if array.ndim == 1:  # For 1D arrays like RGB values
                return torch.tensor(array, device=self.device, dtype=torch.float32)
            elif array.ndim == 3:  # For image data in HWC format
                # Convert from HWC to CHW format
                array = np.transpose(array, (2, 0, 1))
            
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        elif isinstance(array, list):
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        else:
            return array  # Already a tensor or other type
    
    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor  # Already a numpy array or other type
    
    def compute_image_difference(self, target, current):
        """Compute mean squared error between target and current image"""
        return torch.mean((target - current) ** 2).item()
    
    def _clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class CPUAccelerator(Accelerator):
    """CPU accelerator for tensor operations"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
    
    def to_tensor(self, array):
        """Convert numpy array to tensor on CPU"""
        if isinstance(array, np.ndarray):
            # Handle different dimensions properly
            if array.ndim == 1:  # For 1D arrays like RGB values
                return torch.tensor(array, device=self.device, dtype=torch.float32)
            elif array.ndim == 3:  # For image data in HWC format
                # Convert from HWC to CHW format
                array = np.transpose(array, (2, 0, 1))
            
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        elif isinstance(array, list):
            return torch.tensor(array, device=self.device, dtype=torch.float32)
        else:
            return array  # Already a tensor or other type
    
    def to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().numpy()
        return tensor  # Already a numpy array or other type
    
    def compute_image_difference(self, target, current):
        """Compute mean squared error between target and current image"""
        return torch.mean((target - current) ** 2).item() 