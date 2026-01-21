import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    # Get dimensions from tensor shapes
    N = a.shape[0]
    m = N // 2
    
    # This is a convolution pattern that needs to be implemented efficiently
    # Original pattern: a[i] += b[i+m-j-1] * c[j] for all j, i
    
    # Use PyTorch's conv1d for efficient convolution computation
    # We need to reshape for conv1d: input (batch, channels, length)
    
    # Create padded b tensor for proper indexing
    b_padded = torch.zeros(N + m - 1, dtype=b.dtype, device=b.device)
    b_padded[:N] = b
    
    # Build the convolution using matrix multiplication approach
    # For each i, we sum over j: b[i+m-j-1] * c[j]
    # This is equivalent to a matrix-vector multiply
    
    B_matrix = torch.zeros(m, m, dtype=b.dtype, device=b.device)
    
    # Fill the matrix B where B[i, j] = b[i+m-j-1]
    for j in range(m):
        for i in range(m):
            if i + m - j - 1 < N:
                B_matrix[i, j] = b[i + m - j - 1]
    
    # Compute the convolution as matrix multiplication
    result = B_matrix @ c[:m]
    
    # Add to the first m elements of a
    a[:m] += result