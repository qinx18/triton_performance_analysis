import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Create copies for this timestep
        A_copy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        B_copy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Process all blocks for B update
        for block_start in range(1, N-1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N-1)
            
            A_left = tl.load(A_ptr + current_offsets - 1, mask=mask, other=0.0)
            A_center = tl.load(A_ptr + current_offsets, mask=mask, other=0.0)
            A_right = tl.load(A_ptr + current_offsets + 1, mask=mask, other=0.0)
            
            B_new = (1.0/3.0) * (A_left + A_center + A_right)
            tl.store(B_ptr + current_offsets, B_new, mask=mask)
        
        # Process all blocks for A update
        for block_start in range(1, N-1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N-1)
            
            B_left = tl.load(B_ptr + current_offsets - 1, mask=mask, other=0.0)
            B_center = tl.load(B_ptr + current_offsets, mask=mask, other=0.0)
            B_right = tl.load(B_ptr + current_offsets + 1, mask=mask, other=0.0)
            
            A_new = (1.0/3.0) * (B_left + B_center + B_right)
            tl.store(A_ptr + current_offsets, A_new, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    
    grid = (1,)
    
    jacobi_1d_kernel[grid](
        A, B, N, TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )