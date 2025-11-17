"""
2D Laplacian Stencil - Baseline Implementation

5-point stencil for 2D Laplacian operator:
    f[i,j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]

This is a common stencil computation used in:
- Heat equation solvers
- Poisson equation solvers
- Image processing (Laplacian filter)
- Scientific computing

Memory Pattern: 5 reads per output (up, down, left, right, center)
"""

import torch
import torch.nn.functional as F

def laplacian_2d_baseline(u):
    """
    Naive PyTorch baseline for 2D Laplacian using slicing

    Args:
        u: Input tensor of shape (batch, height, width)

    Returns:
        f: Laplacian result of shape (batch, height-2, width-2)
           (interior points only, excluding boundaries)
    """
    # Extract interior point neighbors
    # u[i-1, j] - up
    up = u[:, :-2, 1:-1]
    # u[i+1, j] - down
    down = u[:, 2:, 1:-1]
    # u[i, j-1] - left
    left = u[:, 1:-1, :-2]
    # u[i, j+1] - right
    right = u[:, 1:-1, 2:]
    # u[i, j] - center
    center = u[:, 1:-1, 1:-1]

    # 5-point stencil: up + down + left + right - 4*center
    f = up + down + left + right - 4.0 * center

    return f


def laplacian_2d_conv(u):
    """
    Alternative baseline using conv2d with Laplacian kernel

    This uses PyTorch's highly optimized conv2d operation.
    """
    batch, height, width = u.shape

    # Define Laplacian kernel
    # [ 0  1  0]
    # [ 1 -4  1]
    # [ 0  1  0]
    kernel = torch.tensor([
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]
    ], device=u.device, dtype=u.dtype).reshape(1, 1, 3, 3)

    # Apply convolution
    u_4d = u.unsqueeze(1)  # Add channel dimension
    f = F.conv2d(u_4d, kernel, padding=0)
    f = f.squeeze(1)  # Remove channel dimension

    return f


def laplacian_2d_naive_loop(u):
    """
    Pure loop-based implementation (very slow, for reference only)
    Shows what the stencil operation actually does
    """
    batch, height, width = u.shape
    f = torch.zeros(batch, height-2, width-2, device=u.device, dtype=u.dtype)

    for b in range(batch):
        for i in range(1, height-1):
            for j in range(1, width-1):
                f[b, i-1, j-1] = (u[b, i-1, j] +    # up
                                  u[b, i+1, j] +    # down
                                  u[b, i, j-1] +    # left
                                  u[b, i, j+1] -    # right
                                  4.0 * u[b, i, j]) # center

    return f
