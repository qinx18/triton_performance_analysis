import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute base indices for this block
    base_i = pid_i * BLOCK_SIZE + 1  # Start from 1 to handle interior points
    base_j = pid_j * BLOCK_SIZE + 1
    
    # Create offset vectors once
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Synchronize between phases
        tl.debug_barrier()
        
        # First phase: B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
        for block_i in range(BLOCK_SIZE):
            i = base_i + block_i
            if i < N - 1:  # Check upper bound
                for block_j in range(BLOCK_SIZE):
                    j = base_j + block_j
                    if j < N - 1:  # Check upper bound
                        # Load 5-point stencil from A
                        center = tl.load(A_ptr + i * N + j)
                        left = tl.load(A_ptr + i * N + (j - 1))
                        right = tl.load(A_ptr + i * N + (j + 1))
                        up = tl.load(A_ptr + (i - 1) * N + j)
                        down = tl.load(A_ptr + (i + 1) * N + j)
                        
                        # Compute average and store to B
                        result = 0.2 * (center + left + right + down + up)
                        tl.store(B_ptr + i * N + j, result)
        
        # Synchronize between phases
        tl.debug_barrier()
        
        # Second phase: A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j])
        for block_i in range(BLOCK_SIZE):
            i = base_i + block_i
            if i < N - 1:  # Check upper bound
                for block_j in range(BLOCK_SIZE):
                    j = base_j + block_j
                    if j < N - 1:  # Check upper bound
                        # Load 5-point stencil from B
                        center = tl.load(B_ptr + i * N + j)
                        left = tl.load(B_ptr + i * N + (j - 1))
                        right = tl.load(B_ptr + i * N + (j + 1))
                        up = tl.load(B_ptr + (i - 1) * N + j)
                        down = tl.load(B_ptr + (i + 1) * N + j)
                        
                        # Compute average and store to A
                        result = 0.2 * (center + left + right + down + up)
                        tl.store(A_ptr + i * N + j, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    # Adjust grid to cover interior points (N-2)x(N-2)
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_size, grid_size)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )