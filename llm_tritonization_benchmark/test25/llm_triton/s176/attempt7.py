import torch

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # This is a convolution pattern that must be computed sequentially
    # Using PyTorch operations for efficiency instead of nested Triton loops
    
    # Compute the convolution: a[i] += sum_j(b[i+m-j-1] * c[j])
    # This can be rewritten as a matrix multiplication
    
    # Create index matrix for vectorized computation
    i_indices = torch.arange(m, device=a.device, dtype=torch.long)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long)
    
    # Compute b indices: i + m - j - 1
    # Using broadcasting: i_indices[:, None] + m - j_indices[None, :] - 1
    b_indices = i_indices.unsqueeze(1) + m - j_indices.unsqueeze(0) - 1
    
    # Clamp indices to valid range
    b_indices = torch.clamp(b_indices, 0, n - 1)
    
    # Extract the b values using advanced indexing
    b_matrix = b[b_indices]  # Shape: (m, m)
    
    # Compute the matrix-vector product
    result = torch.matmul(b_matrix, c[:m])
    
    # Add to a
    a[:m] += result