import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, A_copy_ptr, N, t_start, BLOCK_SIZE_I: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    
    # Calculate i range for this block
    i_start = 1 + pid_i * BLOCK_SIZE_I
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i_offsets <= N - 2
    
    # Process all j values for each i in this block
    for i_idx in range(BLOCK_SIZE_I):
        i = i_start + i_idx
        if i <= N - 2:
            for j in range(1, N - 1):
                # Calculate linear indices for 9-point stencil
                idx_center = i * N + j
                
                # Read from copy (to avoid WAR dependencies)
                idx_im1_jm1 = (i - 1) * N + (j - 1)
                idx_im1_j = (i - 1) * N + j
                idx_im1_jp1 = (i - 1) * N + (j + 1)
                idx_i_jm1 = i * N + (j - 1)
                idx_i_j = i * N + j
                idx_i_jp1 = i * N + (j + 1)
                idx_ip1_jm1 = (i + 1) * N + (j - 1)
                idx_ip1_j = (i + 1) * N + j
                idx_ip1_jp1 = (i + 1) * N + (j + 1)
                
                # Load values from copy
                val_im1_jm1 = tl.load(A_copy_ptr + idx_im1_jm1)
                val_im1_j = tl.load(A_copy_ptr + idx_im1_j)
                val_im1_jp1 = tl.load(A_copy_ptr + idx_im1_jp1)
                val_i_jm1 = tl.load(A_copy_ptr + idx_i_jm1)
                val_i_j = tl.load(A_copy_ptr + idx_i_j)
                val_i_jp1 = tl.load(A_copy_ptr + idx_i_jp1)
                val_ip1_jm1 = tl.load(A_copy_ptr + idx_ip1_jm1)
                val_ip1_j = tl.load(A_copy_ptr + idx_ip1_j)
                val_ip1_jp1 = tl.load(A_copy_ptr + idx_ip1_jp1)
                
                # Compute average
                result = (val_im1_jm1 + val_im1_j + val_im1_jp1 +
                         val_i_jm1 + val_i_j + val_i_jp1 +
                         val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
                
                # Store result
                tl.store(A_ptr + idx_center, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Sequential time steps due to dependencies
    for t in range(TSTEPS):
        # Create read-only copy to handle WAR dependencies
        A_copy = A.clone()
        
        # Launch kernel for parallel i computation
        BLOCK_SIZE_I = 32
        grid_size_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
        
        seidel_2d_kernel[(grid_size_i,)](
            A, A_copy, N, t, BLOCK_SIZE_I
        )