import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    
    # Pre-compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            # Load A values
            a_left = tl.load(A + current_offsets - 1, mask=mask, other=0.0)
            a_center = tl.load(A + current_offsets, mask=mask, other=0.0)
            a_right = tl.load(A + current_offsets + 1, mask=mask, other=0.0)
            
            # Compute and store B values
            b_vals = 0.33333 * (a_left + a_center + a_right)
            tl.store(B + current_offsets, b_vals, mask=mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            # Load B values
            b_left = tl.load(B + current_offsets - 1, mask=mask, other=0.0)
            b_center = tl.load(B + current_offsets, mask=mask, other=0.0)
            b_right = tl.load(B + current_offsets + 1, mask=mask, other=0.0)
            
            # Compute and store A values
            a_vals = 0.33333 * (b_left + b_center + b_right)
            tl.store(A + current_offsets, a_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    jacobi_1d_kernel[(1,)](A, B, N, TSTEPS)