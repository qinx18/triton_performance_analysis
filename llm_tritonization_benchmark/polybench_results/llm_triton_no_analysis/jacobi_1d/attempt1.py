import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Mask for valid indices (1 to N-2 inclusive)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    for t in range(TSTEPS):
        # First step: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_left = tl.load(A_ptr + offsets - 1, mask=mask)
        a_center = tl.load(A_ptr + offsets, mask=mask)
        a_right = tl.load(A_ptr + offsets + 1, mask=mask)
        b_new = 0.33333 * (a_left + a_center + a_right)
        tl.store(B_ptr + offsets, b_new, mask=mask)
        
        # Second step: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_left = tl.load(B_ptr + offsets - 1, mask=mask)
        b_center = tl.load(B_ptr + offsets, mask=mask)
        b_right = tl.load(B_ptr + offsets + 1, mask=mask)
        a_new = 0.33333 * (b_left + b_center + b_right)
        tl.store(A_ptr + offsets, a_new, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    jacobi_1d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )