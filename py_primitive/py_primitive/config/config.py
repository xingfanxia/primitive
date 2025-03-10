"""
Configuration management for the GPU-accelerated primitive image generator.
"""
import os

# Default configuration values
DEFAULT_CONFIG = {
    # Image processing
    "input_resize": 256,
    "output_size": 1024,
    
    # Algorithm parameters
    "shape_count": 100,
    "shape_mode": 1,  # 1=triangle, 2=rect, 3=ellipse, 4=circle, 5=rotatedrect
    "shape_alpha": 128,
    
    # Differential Evolution parameters
    "population_size": 50,
    "generations": 20,
    "crossover_probability": 0.7,
    "mutation_factor": 0.8,
    
    # GPU/Performance settings
    "use_gpu": True,
    "batch_size": 32,
    "num_workers": 4,
}

# Shape types
SHAPE_TYPES = {
    0: "combo",
    1: "triangle",
    2: "rectangle",
    3: "ellipse",
    4: "circle",
    5: "rotated_rectangle",
    6: "bezier",
    7: "rotated_ellipse",
    8: "polygon"
}

def get_config(user_config=None):
    """
    Returns configuration with user overrides.
    
    Args:
        user_config (dict): User-provided configuration overrides
        
    Returns:
        dict: Complete configuration with defaults applied
    """
    config = DEFAULT_CONFIG.copy()
    
    # Apply environment variable overrides
    for key in config.keys():
        env_var = f"PRIMITIVE_{key.upper()}"
        if env_var in os.environ:
            # Handle type conversion appropriately
            if isinstance(config[key], bool):
                config[key] = os.environ[env_var].lower() in ('true', 'yes', '1')
            elif isinstance(config[key], int):
                config[key] = int(os.environ[env_var])
            elif isinstance(config[key], float):
                config[key] = float(os.environ[env_var])
            else:
                config[key] = os.environ[env_var]
    
    # Apply user config overrides
    if user_config:
        for key, value in user_config.items():
            if key in config:
                config[key] = value
    
    return config 