import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 2D grid (i, j)
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate row and column indices
    i = pid_i + 1  # Skip boundary (starts from 1)
    j_start = pid_j * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets + 1  # Skip boundary (starts from 1)
    
    # Mask for valid j indices (must be < N-1)
    mask = j_indices < N - 1
    
    # Check if i is valid (must be < N-1)
    if i >= N - 1:
        return
    
    # Time step loop (sequential)
    for t in range(TSTEPS):
        # First phase: compute B from A
        # Current position indices
        center_idx = i * N + j_indices
        left_idx = i * N + (j_indices - 1)
        right_idx = i * N + (j_indices + 1)
        up_idx = (i - 1) * N + j_indices
        down_idx = (i + 1) * N + j_indices
        
        # Load values from A
        center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
        right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
        up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
        down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
        
        # Compute and store to B
        result = 0.2 * (center + left + right + up + down)
        tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Synchronize to ensure all threads complete phase 1
        tl.debug_barrier()
        
        # Second phase: compute A from B
        # Load values from B
        center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
        right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
        up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
        down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
        
        # Compute and store to A
        result = 0.2 * (center + left + right + up + down)
        tl.store(A_ptr + center_idx, result, mask=mask)
        
        # Synchronize to ensure all threads complete phase 2
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    grid_i = N - 2  # Interior points only (1 to N-2)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_i, grid_j)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE=BLOCK_SIZE
    )