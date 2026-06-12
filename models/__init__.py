# Models package
# Only expose the safe function — class imports require torch
from .cnn_attention_module import calculate_cnn_attention_stress

__all__ = ["calculate_cnn_attention_stress"]
