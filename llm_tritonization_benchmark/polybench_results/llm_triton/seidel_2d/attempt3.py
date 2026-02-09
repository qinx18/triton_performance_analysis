import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, A_copy_ptr, N, t, stride_0):
    # Get program IDs for parallelizing over i and j dimensions
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Current i and j indices (offset by 1 since we process from 1 to N-2)
    i = pid_i + 1
    j = pid_j + 1
    
    # Check bounds
    if i >= N - 1 or j >= N - 1:
        return
    
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
    
    # Load values from 9-point stencil (read from copy to avoid WAR)
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
    
    # Store result (write to original)
    tl.store(A_ptr + idx_i_j, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Sequential time steps to maintain dependencies
    for t in range(TSTEPS):
        # Create read-only copy for this time step
        A_copy = A.clone()
        
        # Grid size: parallelize over i and j dimensions
        grid = (N - 2, N - 2)
        
        # Launch kernel
        seidel_2d_kernel[grid](
            A, A_copy, N, t,
            A.stride(0)
        )