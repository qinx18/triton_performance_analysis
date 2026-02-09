import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 2D grid
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate base indices
    i = pid_i + 1  # Interior points start from 1
    j_start = pid_j * BLOCK_SIZE + 1  # Interior points start from 1
    
    # Create offset vector once
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets
    
    # Mask for valid j indices
    mask = j_indices < N - 1
    
    # Check if i is valid
    if i >= N - 1:
        return
    
    # Pre-compute base addresses for this row
    center_base = i * N
    up_base = (i - 1) * N
    down_base = (i + 1) * N
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Phase 1: A -> B
        # Load center and neighbors from A
        center = tl.load(A_ptr + center_base + j_indices, mask=mask, other=0.0)
        left = tl.load(A_ptr + center_base + j_indices - 1, mask=mask, other=0.0)
        right = tl.load(A_ptr + center_base + j_indices + 1, mask=mask, other=0.0)
        up = tl.load(A_ptr + up_base + j_indices, mask=mask, other=0.0)
        down = tl.load(A_ptr + down_base + j_indices, mask=mask, other=0.0)
        
        # Compute stencil and store to B
        result_b = 0.2 * (center + left + right + up + down)
        tl.store(B_ptr + center_base + j_indices, result_b, mask=mask)
        
        # Global synchronization point
        tl.debug_barrier()
        
        # Phase 2: B -> A
        # Load center and neighbors from B
        center = tl.load(B_ptr + center_base + j_indices, mask=mask, other=0.0)
        left = tl.load(B_ptr + center_base + j_indices - 1, mask=mask, other=0.0)
        right = tl.load(B_ptr + center_base + j_indices + 1, mask=mask, other=0.0)
        up = tl.load(B_ptr + up_base + j_indices, mask=mask, other=0.0)
        down = tl.load(B_ptr + down_base + j_indices, mask=mask, other=0.0)
        
        # Compute stencil and store to A
        result_a = 0.2 * (center + left + right + up + down)
        tl.store(A_ptr + center_base + j_indices, result_a, mask=mask)
        
        # Global synchronization point
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    
    # Grid covers interior points only
    grid_i = N - 2  # rows 1 to N-2
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)  # columns 1 to N-2
    
    jacobi_2d_kernel[(grid_i, grid_j)](
        A, B, N, TSTEPS, BLOCK_SIZE=BLOCK_SIZE
    )