import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, x_copy_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get row index
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Initialize x[i] = b[i]
    x_i = tl.load(b_ptr + i)
    
    # Sequential processing for j < i
    for j in range(i):
        # Load L[i][j] and x[j]
        L_ij = tl.load(L_ptr + i * N + j)
        x_j = tl.load(x_copy_ptr + j)
        
        # Subtract L[i][j] * x[j]
        x_i = x_i - L_ij * x_j
    
    # Divide by L[i][i]
    L_diag = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_diag
    
    # Store result
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    # Create read-only copy of x to handle WAR dependencies
    x_copy = x.clone()
    
    # Launch configuration
    grid = (N,)
    BLOCK_SIZE = 64
    
    trisolv_kernel[grid](
        L, b, x, x_copy, N, BLOCK_SIZE
    )