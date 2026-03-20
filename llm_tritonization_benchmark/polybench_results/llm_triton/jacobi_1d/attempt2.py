import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = 128
    
    for t in range(TSTEPS):
        tl.debug_barrier()
        
        # Phase 1: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for i_start in range(1, N - 1, BLOCK_SIZE):
            offsets = i_start + tl.arange(0, BLOCK_SIZE)
            mask = (offsets >= 1) & (offsets < N - 1)
            
            a_prev = tl.load(A_ptr + offsets - 1, mask=mask)
            a_curr = tl.load(A_ptr + offsets, mask=mask)  
            a_next = tl.load(A_ptr + offsets + 1, mask=mask)
            
            b_vals = 0.33333 * (a_prev + a_curr + a_next)
            tl.store(B_ptr + offsets, b_vals, mask=mask)
        
        tl.debug_barrier()
        
        # Phase 2: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])  
        for i_start in range(1, N - 1, BLOCK_SIZE):
            offsets = i_start + tl.arange(0, BLOCK_SIZE)
            mask = (offsets >= 1) & (offsets < N - 1)
            
            b_prev = tl.load(B_ptr + offsets - 1, mask=mask)
            b_curr = tl.load(B_ptr + offsets, mask=mask)
            b_next = tl.load(B_ptr + offsets + 1, mask=mask)
            
            a_vals = 0.33333 * (b_prev + b_curr + b_next)
            tl.store(A_ptr + offsets, a_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    grid = (1,)
    jacobi_1d_kernel[grid](A, B, N, TSTEPS)