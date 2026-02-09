import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time step loop (sequential)
    for t in range(TSTEPS):
        # First phase: compute B from A for all valid (i,j) pairs
        for i in range(1, N - 1):
            j_offsets = block_start + offsets
            mask = j_offsets < N - 1
            mask = mask & (j_offsets >= 1)
            
            # Current position indices
            center_idx = i * N + j_offsets
            left_idx = i * N + (j_offsets - 1)
            right_idx = i * N + (j_offsets + 1)
            up_idx = (i - 1) * N + j_offsets
            down_idx = (i + 1) * N + j_offsets
            
            # Load values
            center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
            left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
            right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
            up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
            down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
            
            # Compute and store
            result = 0.2 * (center + left + right + up + down)
            tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Second phase: compute A from B for all valid (i,j) pairs
        for i in range(1, N - 1):
            j_offsets = block_start + offsets
            mask = j_offsets < N - 1
            mask = mask & (j_offsets >= 1)
            
            # Current position indices
            center_idx = i * N + j_offsets
            left_idx = i * N + (j_offsets - 1)
            right_idx = i * N + (j_offsets + 1)
            up_idx = (i - 1) * N + j_offsets
            down_idx = (i + 1) * N + j_offsets
            
            # Load values
            center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
            left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
            right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
            up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
            down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
            
            # Compute and store
            result = 0.2 * (center + left + right + up + down)
            tl.store(A_ptr + center_idx, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE=BLOCK_SIZE
    )