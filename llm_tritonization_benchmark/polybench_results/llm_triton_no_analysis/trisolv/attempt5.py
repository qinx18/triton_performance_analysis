import torch
import triton
import triton.language as tl

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr):
    # This is a sequential algorithm that must be processed one element at a time
    for i in range(N):
        # x[i] = b[i]
        x_i = tl.load(b_ptr + i)
        
        # for j < i: x[i] -= L[i][j] * x[j]
        for j in range(i):
            L_ij = tl.load(L_ptr + i * N + j)
            x_j = tl.load(x_ptr + j)
            x_i = x_i - L_ij * x_j
        
        # x[i] = x[i] / L[i][i]
        L_ii = tl.load(L_ptr + i * N + i)
        x_i = x_i / L_ii
        
        # Store result
        tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    grid = (1,)  # Single program since this is inherently sequential
    trisolv_kernel[grid](L, b, x, N)