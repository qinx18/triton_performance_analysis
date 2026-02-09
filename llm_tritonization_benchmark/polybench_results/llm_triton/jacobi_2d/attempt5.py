import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    block_id = tl.program_id(0)
    
    # Calculate total interior elements
    interior_size = (N - 2) * (N - 2)
    
    # Calculate starting position for this block
    block_start = block_id * BLOCK_SIZE
    
    # Create offsets for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    mask = element_ids < interior_size
    
    # Convert 1D element_ids to 2D coordinates in interior region
    interior_i = element_ids // (N - 2) + 1
    interior_j = element_ids % (N - 2) + 1
    
    # Perform TSTEPS iterations
    for t in range(TSTEPS):
        # First phase: compute B from A
        center_idx = interior_i * N + interior_j
        left_idx = interior_i * N + (interior_j - 1)
        right_idx = interior_i * N + (interior_j + 1)
        up_idx = (interior_i - 1) * N + interior_j
        down_idx = (interior_i + 1) * N + interior_j
        
        # Load A values for stencil
        a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
        
        # Compute B values
        b_new = 0.2 * (a_center + a_left + a_right + a_down + a_up)
        
        # Store B values
        tl.store(B_ptr + center_idx, b_new, mask=mask)
        
        # Synchronize before second phase
        tl.debug_barrier()
        
        # Second phase: compute A from B
        # Load B values for stencil
        b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
        
        # Compute A values
        a_new = 0.2 * (b_center + b_left + b_right + b_down + b_up)
        
        # Store A values
        tl.store(A_ptr + center_idx, a_new, mask=mask)
        
        # Synchronize before next iteration
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Calculate number of interior elements
    interior_size = (N - 2) * (N - 2)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel
    grid = (num_blocks,)
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )