import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total interior elements
    interior_size = (N - 2) * (N - 2)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Mask for valid elements
    mask = element_ids < interior_size
    
    # Convert 1D element_ids to 2D coordinates (i, j) in interior region
    interior_i = element_ids // (N - 2) + 1
    interior_j = element_ids % (N - 2) + 1
    
    for t in range(TSTEPS):
        # First loop: B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
        
        # Load A values
        a_center = tl.load(A_ptr + interior_i * N + interior_j, mask=mask)
        a_left = tl.load(A_ptr + interior_i * N + (interior_j - 1), mask=mask)
        a_right = tl.load(A_ptr + interior_i * N + (interior_j + 1), mask=mask)
        a_down = tl.load(A_ptr + (interior_i + 1) * N + interior_j, mask=mask)
        a_up = tl.load(A_ptr + (interior_i - 1) * N + interior_j, mask=mask)
        
        # Compute B values
        b_new = 0.2 * (a_center + a_left + a_right + a_down + a_up)
        
        # Store B values
        tl.store(B_ptr + interior_i * N + interior_j, b_new, mask=mask)
        
        # Second loop: A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j])
        
        # Load B values
        b_center = tl.load(B_ptr + interior_i * N + interior_j, mask=mask)
        b_left = tl.load(B_ptr + interior_i * N + (interior_j - 1), mask=mask)
        b_right = tl.load(B_ptr + interior_i * N + (interior_j + 1), mask=mask)
        b_down = tl.load(B_ptr + (interior_i + 1) * N + interior_j, mask=mask)
        b_up = tl.load(B_ptr + (interior_i - 1) * N + interior_j, mask=mask)
        
        # Compute A values
        a_new = 0.2 * (b_center + b_left + b_right + b_down + b_up)
        
        # Store A values
        tl.store(A_ptr + interior_i * N + interior_j, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Calculate total interior elements
    interior_size = (N - 2) * (N - 2)
    
    # Block size
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(num_blocks,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )