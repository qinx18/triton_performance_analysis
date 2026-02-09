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
    
    # Sequential reduction for j < i
    for j_start in range(0, i, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        
        # Mask for valid j indices (j < i)
        j_mask = j_indices < i
        
        # Load L[i][j] values
        L_row_ptr = L_ptr + i * N
        L_vals = tl.load(L_row_ptr + j_indices, mask=j_mask, other=0.0)
        
        # Load x_copy[j] values
        x_vals = tl.load(x_copy_ptr + j_indices, mask=j_mask, other=0.0)
        
        # Compute partial products and sum
        products = L_vals * x_vals
        partial_sum = tl.sum(products)
        
        # Subtract from x_i
        x_i = x_i - partial_sum
    
    # Divide by L[i][i]
    L_diag = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_diag
    
    # Store result
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    # Create read-only copy of x to handle WAR dependencies
    x_copy = x.clone()
    
    # Launch configuration
    BLOCK_SIZE = 64
    grid = (N,)
    
    trisolv_kernel[grid](
        L, b, x, x_copy, N, BLOCK_SIZE
    )