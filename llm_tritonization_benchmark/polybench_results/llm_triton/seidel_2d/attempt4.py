import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, A_copy_ptr, N, stride_0, BLOCK_SIZE: tl.constexpr):
    # Get program IDs for parallelizing over linearized (i,j) space
    pid = tl.program_id(0)
    
    # Calculate block start in linearized space
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    linear_indices = block_start + offsets
    
    # Convert linear indices back to (i,j) coordinates
    # Linear index = (i-1) * (N-2) + (j-1) for interior points
    interior_size = N - 2
    i_interior = linear_indices // interior_size
    j_interior = linear_indices % interior_size
    
    # Actual i,j coordinates (offset by 1 since we process from 1 to N-2)
    i = i_interior + 1
    j = j_interior + 1
    
    # Mask for valid indices
    valid_mask = linear_indices < interior_size * interior_size
    
    # Calculate linear indices for 9-point stencil
    # A[i-1][j-1], A[i-1][j], A[i-1][j+1]
    idx_im1_jm1 = (i - 1) * stride_0 + (j - 1)
    idx_im1_j = (i - 1) * stride_0 + j
    idx_im1_jp1 = (i - 1) * stride_0 + (j + 1)
    
    # A[i][j-1], A[i][j], A[i][j+1]  
    idx_i_jm1 = i * stride_0 + (j - 1)
    idx_i_j = i * stride_0 + j
    idx_i_jp1 = i * stride_0 + (j + 1)
    
    # A[i+1][j-1], A[i+1][j], A[i+1][j+1]
    idx_ip1_jm1 = (i + 1) * stride_0 + (j - 1)
    idx_ip1_j = (i + 1) * stride_0 + j
    idx_ip1_jp1 = (i + 1) * stride_0 + (j + 1)
    
    # Load values from 9-point stencil (read from copy)
    val_im1_jm1 = tl.load(A_copy_ptr + idx_im1_jm1, mask=valid_mask)
    val_im1_j = tl.load(A_copy_ptr + idx_im1_j, mask=valid_mask)
    val_im1_jp1 = tl.load(A_copy_ptr + idx_im1_jp1, mask=valid_mask)
    
    val_i_jm1 = tl.load(A_copy_ptr + idx_i_jm1, mask=valid_mask)
    val_i_j = tl.load(A_copy_ptr + idx_i_j, mask=valid_mask)
    val_i_jp1 = tl.load(A_copy_ptr + idx_i_jp1, mask=valid_mask)
    
    val_ip1_jm1 = tl.load(A_copy_ptr + idx_ip1_jm1, mask=valid_mask)
    val_ip1_j = tl.load(A_copy_ptr + idx_ip1_j, mask=valid_mask)
    val_ip1_jp1 = tl.load(A_copy_ptr + idx_ip1_jp1, mask=valid_mask)
    
    # Compute average
    result = (val_im1_jm1 + val_im1_j + val_im1_jp1 +
             val_i_jm1 + val_i_j + val_i_jp1 +
             val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
    
    # Store result (write to original)
    tl.store(A_ptr + idx_i_j, result, mask=valid_mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    
    # Sequential time steps to maintain dependencies
    for t in range(TSTEPS):
        # Create read-only copy for this time step
        A_copy = A.clone()
        
        # Total number of interior points
        interior_size = N - 2
        total_points = interior_size * interior_size
        
        # Grid size
        grid = (triton.cdiv(total_points, BLOCK_SIZE),)
        
        # Launch kernel
        seidel_2d_kernel[grid](
            A, A_copy, N, 
            A.stride(0), BLOCK_SIZE
        )