import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, L_stride0):
    # Get program ID to identify which iteration this handles
    i = tl.program_id(0)
    
    if i < N:
        # Load b[i] into x[i]
        b_val = tl.load(b_ptr + i)
        x_val = b_val
        
        # Compute the dot product for j < i
        if i > 0:
            # Process in chunks of BLOCK_SIZE
            BLOCK_SIZE = 32
            j_offsets = tl.arange(0, BLOCK_SIZE)
            
            for j_start in range(0, i, BLOCK_SIZE):
                current_j = j_start + j_offsets
                j_mask = current_j < i
                
                # Load L[i][j] values
                L_indices = i * L_stride0 + current_j
                L_vals = tl.load(L_ptr + L_indices, mask=j_mask, other=0.0)
                
                # Load x[j] values
                x_vals = tl.load(x_ptr + current_j, mask=j_mask, other=0.0)
                
                # Compute partial dot product
                products = L_vals * x_vals
                partial_sum = tl.sum(products)
                x_val = x_val - partial_sum
        
        # Divide by diagonal element L[i][i]
        diag_val = tl.load(L_ptr + i * L_stride0 + i)
        result = x_val / diag_val
        tl.store(x_ptr + i, result)

def trisolv_triton(L, b, x, N):
    # Copy b to x initially
    x.copy_(b)
    
    # Launch one thread per row, but process sequentially
    for i in range(N):
        grid = (1,)
        trisolv_kernel[grid](
            L, b, x, i + 1, L.stride(0)
        )