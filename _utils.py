""" 
Sparsity-Promoting Dynamic Mode Decomposition 
via Sequentially Thresholded Quadratic Programming 
"""

import numpy as np


def count_nnz(x, eps=1e-6):
    return np.sum(np.abs(x) > eps)


def compute_energy_rank(X, energy_threshold=0.99):
    """
    Determine rank by capturing >99% (or custom threshold) of energy in singular values.
    
    Args:
        X (ndarray): Input data matrix (n_features x n_samples)
        energy_threshold (float): Threshold for energy retention (default: 0.99)
    
    Returns:
        int: Optimal rank `r`
    """
    # Compute SVD
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Calculate cumulative energy
    energy = np.cumsum(s**2) / np.sum(s**2)  # Normalized cumulative energy
    
    # Find the smallest rank capturing >99% energy
    r = np.argmax(energy >= energy_threshold) + 1  # +1 for 0-based index
    
    print(f"Singular values: {s[:10]}...")  # Print first 10 singular values
    print(f"Cumulative energy at rank {r}: {energy[r-1]:.4f}")
    
    return r


def compute_signal2noise(signal, noise):
    """
    Compute the Signal-to-Noise Ratio (SNR) in decibels (dB).
    
    Parameters:
        signal (array-like): Signal values (can be time series, image, etc.)
        noise (array-like): Noise values (must have same shape as signal)
        
    Returns:
        float: SNR in dB
    """
    # Convert inputs to numpy arrays for vectorized operations
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    
    # Verify shapes match
    if signal.shape != noise.shape:
        raise ValueError("Signal and noise must have the same shape")
    
    # Calculate power of signal and noise
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Handle division by zero (if noise is zero)
    if noise_power == 0:
        return float('inf')  # Infinite SNR if no noise exists
    
    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def add_salt_pepper_noise(u, corruption_rate=0.05, noise_value=10):
    """
    Adds salt-and-pepper noise to a 2D velocity field (e.g., u ∈ ℝ^{m×n}).

    Parameters:
        u: np.ndarray of shape (m, n) – original velocity data
        corruption_rate: float – fraction of entries to corrupt (e.g., 0.01 = 1%)

    Returns:
        u_noisy: np.ndarray – corrupted version of u
    """
    m, n = u.shape
    total_points = m * n
    num_corrupted = int(corruption_rate * total_points)

    # Flattened random indices for corruption
    flat_indices = np.random.choice(total_points, size=num_corrupted, replace=False)
    i, j = np.unravel_index(flat_indices, (m, n))

    # Compute standard deviation of the input
    u_std = np.std(u)

    # Generate noise
    noise = noise_value * u_std * np.random.choice([-1, 1], size=num_corrupted)

    # Apply noise
    u_noisy = u.copy()
    u_noisy[i, j] = noise
    return u_noisy


def compute_prediction_error(Y_pred, Y_true, p=2):
    return np.linalg.norm(Y_pred - Y_true, axis=0, ord=p)/np.linalg.norm(Y_true, axis=0, ord=p)
