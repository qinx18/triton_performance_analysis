import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, A_copy_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting position
    start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask
    mask = offsets < (N - 2) * (N - 2)
    
    # Convert linear index to 2D coordinates (add 1 to account for boundary)
    i = (offsets // (N - 2)) + 1
    j = (offsets % (N - 2)) + 1
    
    # Calculate linear indices for 9-point stencil
    idx_center = i * N + j
    
    # Read all 9 stencil points from copy
    val_im1_jm1 = tl.load(A_copy_ptr + (i - 1) * N + (j - 1), mask=mask)
    val_im1_j = tl.load(A_copy_ptr + (i - 1) * N + j, mask=mask)
    val_im1_jp1 = tl.load(A_copy_ptr + (i - 1) * N + (j + 1), mask=mask)
    val_i_jm1 = tl.load(A_copy_ptr + i * N + (j - 1), mask=mask)
    val_i_j = tl.load(A_copy_ptr + idx_center, mask=mask)
    val_i_jp1 = tl.load(A_copy_ptr + i * N + (j + 1), mask=mask)
    val_ip1_jm1 = tl.load(A_copy_ptr + (i + 1) * N + (j - 1), mask=mask)
    val_ip1_j = tl.load(A_copy_ptr + (i + 1) * N + j, mask=mask)
    val_ip1_jp1 = tl.load(A_copy_ptr + (i + 1) * N + (j + 1), mask=mask)
    
    # Compute average
    result = (val_im1_jm1 + val_im1_j + val_im1_jp1 +
              val_i_jm1 + val_i_j + val_i_jp1 +
              val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
    
    # Store result
    tl.store(A_ptr + idx_center, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    # Sequential time steps due to dependencies
    for t in range(TSTEPS):
        # Create read-only copy to handle WAR dependencies
        A_copy = A.clone()
        
        # Launch kernel for parallel computation
        BLOCK_SIZE = 256
        num_elements = (N - 2) * (N - 2)
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        seidel_2d_kernel[(grid_size,)](
            A, A_copy, N, BLOCK_SIZE
        )