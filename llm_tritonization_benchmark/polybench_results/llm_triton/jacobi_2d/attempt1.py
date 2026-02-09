import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the i dimension
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Time step loop (sequential)
    for t in range(TSTEPS):
        # First phase: compute B from A
        for j in range(1, N - 1):
            # Mask for valid i indices (1 <= i < N-1)
            mask = (i_offsets >= 1) & (i_offsets < N - 1)
            
            # Load A values for the stencil computation
            # A[i][j]
            center_idx = i_offsets * N + j
            center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
            
            # A[i][j-1]
            left_idx = i_offsets * N + (j - 1)
            left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
            
            # A[i][j+1]
            right_idx = i_offsets * N + (j + 1)
            right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
            
            # A[i+1][j]
            down_idx = (i_offsets + 1) * N + j
            down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
            
            # A[i-1][j]
            up_idx = (i_offsets - 1) * N + j
            up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
            
            # Compute B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
            result = 0.2 * (center + left + right + down + up)
            
            # Store to B
            b_idx = i_offsets * N + j
            tl.store(B_ptr + b_idx, result, mask=mask)
        
        # Second phase: compute A from B
        for j in range(1, N - 1):
            # Mask for valid i indices (1 <= i < N-1)
            mask = (i_offsets >= 1) & (i_offsets < N - 1)
            
            # Load B values for the stencil computation
            # B[i][j]
            center_idx = i_offsets * N + j
            center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
            
            # B[i][j-1]
            left_idx = i_offsets * N + (j - 1)
            left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
            
            # B[i][j+1]
            right_idx = i_offsets * N + (j + 1)
            right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
            
            # B[i+1][j]
            down_idx = (i_offsets + 1) * N + j
            down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
            
            # B[i-1][j]
            up_idx = (i_offsets - 1) * N + j
            up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
            
            # Compute A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j])
            result = 0.2 * (center + left + right + down + up)
            
            # Store to A
            a_idx = i_offsets * N + j
            tl.store(A_ptr + a_idx, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Block size for parallelization
    BLOCK_SIZE = 32
    
    # Number of blocks needed for the i dimension (range 1 to N-1)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE=BLOCK_SIZE
    )