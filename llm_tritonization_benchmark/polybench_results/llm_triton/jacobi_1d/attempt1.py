import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            A_prev = tl.load(A_ptr + current_offsets - 1, mask=mask)
            A_curr = tl.load(A_ptr + current_offsets, mask=mask)
            A_next = tl.load(A_ptr + current_offsets + 1, mask=mask)
            
            B_val = 0.33333 * (A_prev + A_curr + A_next)
            tl.store(B_ptr + current_offsets, B_val, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            B_prev = tl.load(B_ptr + current_offsets - 1, mask=mask)
            B_curr = tl.load(B_ptr + current_offsets, mask=mask)
            B_next = tl.load(B_ptr + current_offsets + 1, mask=mask)
            
            A_val = 0.33333 * (B_prev + B_curr + B_next)
            tl.store(A_ptr + current_offsets, A_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    
    grid = (1,)
    jacobi_1d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)