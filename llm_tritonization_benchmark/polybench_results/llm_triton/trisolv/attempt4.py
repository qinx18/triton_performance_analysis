import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, L_stride0):
    i = tl.program_id(0)
    
    if i < N:
        b_val = tl.load(b_ptr + i)
        x_val = b_val
        
        if i > 0:
            BLOCK_SIZE = 32
            j_offsets = tl.arange(0, BLOCK_SIZE)
            
            j_start = 0
            while j_start < i:
                current_j = j_start + j_offsets
                j_mask = current_j < i
                
                L_indices = i * L_stride0 + current_j
                L_vals = tl.load(L_ptr + L_indices, mask=j_mask, other=0.0)
                
                x_vals = tl.load(x_ptr + current_j, mask=j_mask, other=0.0)
                
                products = L_vals * x_vals
                partial_sum = tl.sum(products)
                x_val = x_val - partial_sum
                
                j_start = j_start + BLOCK_SIZE
        
        diag_val = tl.load(L_ptr + i * L_stride0 + i)
        result = x_val / diag_val
        tl.store(x_ptr + i, result)

def trisolv_triton(L, b, x, N):
    x.copy_(b)
    
    for i in range(N):
        grid = (1,)
        trisolv_kernel[grid](
            L, b, x, i + 1, L.stride(0)
        )