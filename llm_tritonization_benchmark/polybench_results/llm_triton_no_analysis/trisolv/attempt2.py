import torch
import triton
import triton.language as tl

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, i: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # x[i] = b[i]
    x_val = tl.load(b_ptr + i)
    
    # for (j = 0; j < i; j++)
    for j in range(i):
        L_val = tl.load(L_ptr + i * N + j)
        x_j = tl.load(x_ptr + j)
        x_val = x_val - L_val * x_j
    
    # x[i] = x[i] / L[i][i]
    L_diag = tl.load(L_ptr + i * N + i)
    x_val = x_val / L_diag
    
    tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    for i in range(N):
        grid = (1,)
        trisolv_kernel[grid](L, b, x, i, N)