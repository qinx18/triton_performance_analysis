import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_phase1_kernel(A, B, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get flat index for this thread block
    pid = tl.program_id(0)
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Convert flat index to 2D coordinates (skip boundaries)
    total_elements = (N - 2) * (N - 2)
    mask = flat_idx < total_elements
    
    # Recover i, j coordinates in the interior region [1, N-1)
    j_interior = flat_idx % (N - 2)
    i_interior = flat_idx // (N - 2)
    
    # Convert to actual array indices
    i = i_interior + 1
    j = j_interior + 1
    
    # Load 5-point stencil from A
    center = tl.load(A + i * N + j, mask=mask)
    left = tl.load(A + i * N + (j - 1), mask=mask)
    right = tl.load(A + i * N + (j + 1), mask=mask)
    up = tl.load(A + (i - 1) * N + j, mask=mask)
    down = tl.load(A + (i + 1) * N + j, mask=mask)
    
    # Compute B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
    result = 0.2 * (center + left + right + down + up)
    
    # Store to B
    tl.store(B + i * N + j, result, mask=mask)

@triton.jit
def jacobi_2d_phase2_kernel(A, B, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get flat index for this thread block
    pid = tl.program_id(0)
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Convert flat index to 2D coordinates (skip boundaries)
    total_elements = (N - 2) * (N - 2)
    mask = flat_idx < total_elements
    
    # Recover i, j coordinates in the interior region [1, N-1)
    j_interior = flat_idx % (N - 2)
    i_interior = flat_idx // (N - 2)
    
    # Convert to actual array indices
    i = i_interior + 1
    j = j_interior + 1
    
    # Load 5-point stencil from B
    center = tl.load(B + i * N + j, mask=mask)
    left = tl.load(B + i * N + (j - 1), mask=mask)
    right = tl.load(B + i * N + (j + 1), mask=mask)
    up = tl.load(B + (i - 1) * N + j, mask=mask)
    down = tl.load(B + (i + 1) * N + j, mask=mask)
    
    # Compute A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j])
    result = 0.2 * (center + left + right + down + up)
    
    # Store to A
    tl.store(A + i * N + j, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    total_elements = (N - 2) * (N - 2)  # Interior elements only
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    for t in range(TSTEPS):
        # Phase 1: Update B based on A
        jacobi_2d_phase1_kernel[grid](A, B, N, BLOCK_SIZE)
        
        # Phase 2: Update A based on B
        jacobi_2d_phase2_kernel[grid](A, B, N, BLOCK_SIZE)

def jacobi_2d_kernel():
    jacobi_2d_triton(A, B, N, TSTEPS)