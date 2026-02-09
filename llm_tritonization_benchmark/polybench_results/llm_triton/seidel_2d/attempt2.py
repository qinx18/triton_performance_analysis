import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, stride_0):
    # Get program IDs for parallelizing over i dimension
    pid_i = tl.program_id(0)
    
    # Current i index (offset by 1 since we process i from 1 to N-2)
    i = pid_i + 1
    
    # Check if this thread should process this i
    if i >= N - 1:
        return
    
    # Sequential time steps
    for t in range(TSTEPS):
        # Process all j values for this i sequentially to maintain dependencies
        for j in range(1, N - 1):
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
            
            # Load values from 9-point stencil
            val_im1_jm1 = tl.load(A_ptr + idx_im1_jm1)
            val_im1_j = tl.load(A_ptr + idx_im1_j)
            val_im1_jp1 = tl.load(A_ptr + idx_im1_jp1)
            
            val_i_jm1 = tl.load(A_ptr + idx_i_jm1)
            val_i_j = tl.load(A_ptr + idx_i_j)
            val_i_jp1 = tl.load(A_ptr + idx_i_jp1)
            
            val_ip1_jm1 = tl.load(A_ptr + idx_ip1_jm1)
            val_ip1_j = tl.load(A_ptr + idx_ip1_j)
            val_ip1_jp1 = tl.load(A_ptr + idx_ip1_jp1)
            
            # Compute average
            result = (val_im1_jm1 + val_im1_j + val_im1_jp1 +
                     val_i_jm1 + val_i_j + val_i_jp1 +
                     val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
            
            # Store result
            tl.store(A_ptr + idx_i_j, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Grid size: parallelize over i dimension (N-2 interior points)
    grid = (N - 2,)
    
    # Launch kernel
    seidel_2d_kernel[grid](
        A, N, TSTEPS,
        A.stride(0)
    )