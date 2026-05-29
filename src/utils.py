import yaml
import torch
from pathlib import Path
from src.exceptions import ConfigurationError

def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML: {e}")
    return config


def get_model_info(model: torch.nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_class": model.__class__.__name__
    }
    return info