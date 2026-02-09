import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A, B, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = 1 + pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < N - 1
    
    for t in range(TSTEPS):
        # Synchronize across all blocks before each phase
        tl.debug_barrier()
        
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_left = tl.load(A + current_offsets - 1, mask=mask, other=0.0)
        a_center = tl.load(A + current_offsets, mask=mask, other=0.0)
        a_right = tl.load(A + current_offsets + 1, mask=mask, other=0.0)
        
        b_vals = 0.33333 * (a_left + a_center + a_right)
        tl.store(B + current_offsets, b_vals, mask=mask)
        
        # Synchronize across all blocks before second phase
        tl.debug_barrier()
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_left = tl.load(B + current_offsets - 1, mask=mask, other=0.0)
        b_center = tl.load(B + current_offsets, mask=mask, other=0.0)
        b_right = tl.load(B + current_offsets + 1, mask=mask, other=0.0)
        
        a_vals = 0.33333 * (b_left + b_center + b_right)
        tl.store(A + current_offsets, a_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    jacobi_1d_kernel[(grid_size,)](A, B, N, TSTEPS, BLOCK_SIZE)