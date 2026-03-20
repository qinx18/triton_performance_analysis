import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    # Sequential execution - no parallelization due to WAR dependencies
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Load 9-point stencil values
                val_nw = tl.load(A_ptr + (i - 1) * N + (j - 1))
                val_n = tl.load(A_ptr + (i - 1) * N + j)
                val_ne = tl.load(A_ptr + (i - 1) * N + (j + 1))
                val_w = tl.load(A_ptr + i * N + (j - 1))
                val_c = tl.load(A_ptr + i * N + j)
                val_e = tl.load(A_ptr + i * N + (j + 1))
                val_sw = tl.load(A_ptr + (i + 1) * N + (j - 1))
                val_s = tl.load(A_ptr + (i + 1) * N + j)
                val_se = tl.load(A_ptr + (i + 1) * N + (j + 1))
                
                # Compute average and store
                result = (val_nw + val_n + val_ne + val_w + val_c + val_e + val_sw + val_s + val_se) / 9.0
                tl.store(A_ptr + i * N + j, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Single CTA execution due to WAR dependencies
    grid = (1,)
    seidel_2d_kernel[grid](A, N, TSTEPS)