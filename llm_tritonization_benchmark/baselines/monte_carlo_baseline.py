"""
Monte Carlo Pi Estimation - Baseline Implementation

Classic Monte Carlo method to estimate π by random sampling:
- Generate random points (x, y) in [0, 1] × [0, 1]
- Count how many fall inside unit circle (x² + y² ≤ 1)
- π ≈ 4 × (points_inside / total_points)

This is a simple but representative Monte Carlo simulation that tests:
- Random number generation on GPU
- Element-wise computation
- Reduction operations

Time Complexity: O(N) where N is number of samples
Space Complexity: O(N)

Use cases:
- Teaching example for Monte Carlo methods
- Benchmarking random number generation
- Testing parallel reduction patterns
"""

import torch


def monte_carlo_pi_baseline(n_samples):
    """
    Estimate π using Monte Carlo method with PyTorch

    Algorithm:
    1. Generate n_samples random points (x, y) in unit square [0,1]×[0,1]
    2. Check if each point is inside unit circle: x² + y² ≤ 1
    3. Ratio of points inside circle approximates π/4
    4. Return estimate: π ≈ 4 × (inside_count / total_count)

    Args:
        n_samples: Number of random samples to generate

    Returns:
        Estimated value of π
    """
    # Generate random points in [0, 1] × [0, 1]
    x = torch.rand(n_samples, device='cuda', dtype=torch.float32)
    y = torch.rand(n_samples, device='cuda', dtype=torch.float32)

    # Check if points are inside unit circle
    inside_circle = (x * x + y * y) <= 1.0

    # Count points inside circle
    inside_count = inside_circle.sum().item()

    # Estimate pi
    pi_estimate = 4.0 * inside_count / n_samples

    return pi_estimate


def monte_carlo_pi_baseline_batched(n_samples, batch_size=1000000):
    """
    Batched version to handle very large sample counts
    Processes samples in batches to avoid memory issues

    Args:
        n_samples: Total number of samples
        batch_size: Number of samples per batch

    Returns:
        Estimated value of π
    """
    total_inside = 0
    remaining = n_samples

    while remaining > 0:
        current_batch = min(batch_size, remaining)

        x = torch.rand(current_batch, device='cuda', dtype=torch.float32)
        y = torch.rand(current_batch, device='cuda', dtype=torch.float32)

        inside_circle = (x * x + y * y) <= 1.0
        total_inside += inside_circle.sum().item()

        remaining -= current_batch

    pi_estimate = 4.0 * total_inside / n_samples
    return pi_estimate


def monte_carlo_pi_cpu(n_samples):
    """
    CPU version for comparison
    Typically much slower due to lack of parallelism
    """
    import random
    import math

    inside_count = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_count += 1

    return 4.0 * inside_count / n_samples
