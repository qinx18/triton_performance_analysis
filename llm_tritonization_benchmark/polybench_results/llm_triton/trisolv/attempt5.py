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
            sum_val = 0.0
            for j in range(i):
                L_val = tl.load(L_ptr + i * L_stride0 + j)
                x_val_j = tl.load(x_ptr + j)
                sum_val += L_val * x_val_j
            x_val = x_val - sum_val
        
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