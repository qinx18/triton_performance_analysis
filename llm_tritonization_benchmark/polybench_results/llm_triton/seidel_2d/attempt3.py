import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, A_copy_ptr, N, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block starting positions
    i_start = 1 + pid_i * BLOCK_SIZE_I
    j_start = 1 + pid_j * BLOCK_SIZE_J
    
    # Create offset arrays
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    # Create masks
    i_mask = i_offsets <= N - 2
    j_mask = j_offsets <= N - 2
    
    # Create 2D indices for vectorized computation
    i_expanded = i_offsets[:, None]
    j_expanded = j_offsets[None, :]
    
    # Create 2D mask
    mask_2d = i_mask[:, None] & j_mask[None, :]
    
    # Calculate linear indices for 9-point stencil
    idx_center = i_expanded * N + j_expanded
    
    # Read all 9 stencil points from copy
    val_im1_jm1 = tl.load(A_copy_ptr + (i_expanded - 1) * N + (j_expanded - 1), mask=mask_2d)
    val_im1_j = tl.load(A_copy_ptr + (i_expanded - 1) * N + j_expanded, mask=mask_2d)
    val_im1_jp1 = tl.load(A_copy_ptr + (i_expanded - 1) * N + (j_expanded + 1), mask=mask_2d)
    val_i_jm1 = tl.load(A_copy_ptr + i_expanded * N + (j_expanded - 1), mask=mask_2d)
    val_i_j = tl.load(A_copy_ptr + idx_center, mask=mask_2d)
    val_i_jp1 = tl.load(A_copy_ptr + i_expanded * N + (j_expanded + 1), mask=mask_2d)
    val_ip1_jm1 = tl.load(A_copy_ptr + (i_expanded + 1) * N + (j_expanded - 1), mask=mask_2d)
    val_ip1_j = tl.load(A_copy_ptr + (i_expanded + 1) * N + j_expanded, mask=mask_2d)
    val_ip1_jp1 = tl.load(A_copy_ptr + (i_expanded + 1) * N + (j_expanded + 1), mask=mask_2d)
    
    # Compute average
    result = (val_im1_jm1 + val_im1_j + val_im1_jp1 +
              val_i_jm1 + val_i_j + val_i_jp1 +
              val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
    
    # Store result
    tl.store(A_ptr + idx_center, result, mask=mask_2d)

def seidel_2d_triton(A, N, TSTEPS):
    # Sequential time steps due to dependencies
    for t in range(TSTEPS):
        # Create read-only copy to handle WAR dependencies
        A_copy = A.clone()
        
        # Launch kernel for parallel i,j computation
        BLOCK_SIZE_I = 16
        BLOCK_SIZE_J = 16
        grid_size_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
        grid_size_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
        
        seidel_2d_kernel[(grid_size_i, grid_size_j)](
            A, A_copy, N, BLOCK_SIZE_I, BLOCK_SIZE_J
        )