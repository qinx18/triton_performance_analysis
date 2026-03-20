import torch
import triton
import triton.language as tl

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr):
    # Sequential solver - each row depends on previous results
    for i in range(N):
        # x[i] = b[i]
        b_val = tl.load(b_ptr + i)
        x_val = b_val
        
        # for (j = 0; j < i; j++) x[i] -= L[i][j] * x[j]
        for j in range(i):
            L_ij = tl.load(L_ptr + i * N + j)
            x_j = tl.load(x_ptr + j)
            x_val -= L_ij * x_j
        
        # x[i] = x[i] / L[i][i]
        L_ii = tl.load(L_ptr + i * N + i)
        x_val = x_val / L_ii
        
        # Store result
        tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    # Sequential kernel - single thread block
    grid = (1,)
    trisolv_kernel[grid](L, b, x, N)