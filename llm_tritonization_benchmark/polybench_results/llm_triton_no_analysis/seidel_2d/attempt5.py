import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    # Sequential processing for Gauss-Seidel iteration
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Calculate linear index for current position
                idx = i * N + j
                
                # Load 9-point stencil values
                val_tl = tl.load(A_ptr + (i - 1) * N + (j - 1))
                val_t = tl.load(A_ptr + (i - 1) * N + j)
                val_tr = tl.load(A_ptr + (i - 1) * N + (j + 1))
                val_l = tl.load(A_ptr + i * N + (j - 1))
                val_c = tl.load(A_ptr + i * N + j)
                val_r = tl.load(A_ptr + i * N + (j + 1))
                val_bl = tl.load(A_ptr + (i + 1) * N + (j - 1))
                val_b = tl.load(A_ptr + (i + 1) * N + j)
                val_br = tl.load(A_ptr + (i + 1) * N + (j + 1))
                
                # Compute average
                result = (val_tl + val_t + val_tr + val_l + val_c + val_r + val_bl + val_b + val_br) / 9.0
                
                # Store result
                tl.store(A_ptr + idx, result)

def seidel_2d_triton(A, N, TSTEPS):
    A = A.contiguous()
    
    # Launch single thread kernel for sequential Gauss-Seidel
    grid = (1,)
    
    seidel_2d_kernel[grid](
        A, N, TSTEPS
    )
    
    return A