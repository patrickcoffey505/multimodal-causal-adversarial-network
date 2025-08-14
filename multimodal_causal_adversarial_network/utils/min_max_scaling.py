import tensorflow as tf

def scale_to_unit_range(signal, axis=1, eps=1e-8):
    """
    Normalize EEG/fMRI signal to [-1, 1] per sample (and optionally per channel).
    
    Args:
        signal: Tensor of shape [batch, time, channels]
        axis: Axis to normalize over (usually time axis = 1)
        eps: Small constant to avoid division by zero

    Returns:
        normalized: Tensor in [-1, 1]
        scale_params: Dict with min and max for inversion
    """
    min_val = tf.reduce_min(signal, axis=axis, keepdims=True)
    max_val = tf.reduce_max(signal, axis=axis, keepdims=True)
    
    # Scale to [0, 1]
    norm_01 = (signal - min_val) / (max_val - min_val + eps)
    
    # Scale to [-1, 1]
    normalized = norm_01 * 2.0 - 1.0

    scale_params = {
        "min": min_val,
        "max": max_val
    }
    
    return normalized, scale_params

def inverse_scale_from_unit_range(normalized, scale_params, axis=1):
    """
    Invert normalization from [-1, 1] back to original scale.

    Args:
        normalized: Tensor in [-1, 1]
        scale_params: Dict with 'min' and 'max' from original signal
        axis: Axis that was normalized over

    Returns:
        Original scale signal
    """
    norm_01 = (normalized + 1.0) / 2.0
    original = norm_01 * (scale_params["max"] - scale_params["min"]) + scale_params["min"]
    return original

