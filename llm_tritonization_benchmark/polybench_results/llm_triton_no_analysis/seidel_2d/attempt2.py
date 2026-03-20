import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    # Get program IDs for 2D grid
    i = tl.program_id(0) + 1  # i ranges from 1 to N-2
    j = tl.program_id(1) + 1  # j ranges from 1 to N-2
    
    # Check bounds
    if i >= N - 1:
        return
    if j >= N - 1:
        return
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Calculate linear indices for 9-point stencil
        idx_center = i * N + j
        idx_top_left = (i - 1) * N + (j - 1)
        idx_top = (i - 1) * N + j
        idx_top_right = (i - 1) * N + (j + 1)
        idx_left = i * N + (j - 1)
        idx_right = i * N + (j + 1)
        idx_bottom_left = (i + 1) * N + (j - 1)
        idx_bottom = (i + 1) * N + j
        idx_bottom_right = (i + 1) * N + (j + 1)
        
        # Load all 9 stencil values
        val_top_left = tl.load(A_ptr + idx_top_left)
        val_top = tl.load(A_ptr + idx_top)
        val_top_right = tl.load(A_ptr + idx_top_right)
        val_left = tl.load(A_ptr + idx_left)
        val_center = tl.load(A_ptr + idx_center)
        val_right = tl.load(A_ptr + idx_right)
        val_bottom_left = tl.load(A_ptr + idx_bottom_left)
        val_bottom = tl.load(A_ptr + idx_bottom)
        val_bottom_right = tl.load(A_ptr + idx_bottom_right)
        
        # Compute average of 9 values
        result = (val_top_left + val_top + val_top_right +
                 val_left + val_center + val_right +
                 val_bottom_left + val_bottom + val_bottom_right) / 9.0
        
        # Store result back
        tl.store(A_ptr + idx_center, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Ensure A is contiguous
    A = A.contiguous()
    
    # Launch 2D grid covering interior points [1, N-2] x [1, N-2]
    grid = (N - 2, N - 2)
    
    # Launch kernel
    seidel_2d_kernel[grid](
        A, N, TSTEPS
    )
    
    return A