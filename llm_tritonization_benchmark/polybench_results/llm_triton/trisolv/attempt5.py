import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, x_copy_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for parallelizing over i
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Initialize x[i] = b[i]
    x_i = tl.load(b_ptr + i)
    
    # Process j < i in blocks
    for j_start in range(0, i, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        
        # Mask for valid j indices (j < i)
        j_mask = j_indices < i
        
        # Load L[i][j] values
        L_ptrs = L_ptr + i * N + j_indices
        L_vals = tl.load(L_ptrs, mask=j_mask, other=0.0)
        
        # Load x[j] values from copy
        x_ptrs = x_copy_ptr + j_indices
        x_vals = tl.load(x_ptrs, mask=j_mask, other=0.0)
        
        # Compute L[i][j] * x[j] and accumulate
        products = L_vals * x_vals
        reduction = tl.sum(products)
        x_i = x_i - reduction
    
    # Divide by L[i][i]
    L_diag = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_diag
    
    # Store result
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    # Create read-only copy of x to handle WAR dependencies
    x_copy = x.clone()
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, 1),)
    
    trisolv_kernel[grid](
        L, b, x, x_copy, N, BLOCK_SIZE
    )