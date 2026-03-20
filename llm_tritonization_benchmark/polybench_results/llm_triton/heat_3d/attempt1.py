import torch
import triton
import triton.language as tl

@triton.jit
def heat_3d_phase1_kernel(A, B, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles a block of elements from the 3D space
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear index to 3D coordinates (i, j, k)
    # For interior points: i,j,k in [1, N-2]
    interior_size = N - 2
    total_interior = interior_size * interior_size * interior_size
    
    mask = offsets < total_interior
    
    # Convert linear offset to 3D coordinates
    temp = offsets
    k = temp % interior_size + 1  # k in [1, N-2]
    temp = temp // interior_size
    j = temp % interior_size + 1  # j in [1, N-2]
    i = temp // interior_size + 1  # i in [1, N-2]
    
    # Calculate 3D array indices
    idx = i * N * N + j * N + k
    
    # Load central values
    A_center = tl.load(A + idx, mask=mask)
    
    # Load neighbor values for stencil computation
    A_i_plus = tl.load(A + idx + N * N, mask=mask)   # A[i+1][j][k]
    A_i_minus = tl.load(A + idx - N * N, mask=mask)  # A[i-1][j][k]
    A_j_plus = tl.load(A + idx + N, mask=mask)       # A[i][j+1][k]
    A_j_minus = tl.load(A + idx - N, mask=mask)      # A[i][j-1][k]
    A_k_plus = tl.load(A + idx + 1, mask=mask)       # A[i][j][k+1]
    A_k_minus = tl.load(A + idx - 1, mask=mask)      # A[i][j][k-1]
    
    # Compute stencil: 0.125 * second derivative + original value
    result = (0.125 * (A_i_plus - 2.0 * A_center + A_i_minus) +
              0.125 * (A_j_plus - 2.0 * A_center + A_j_minus) +
              0.125 * (A_k_plus - 2.0 * A_center + A_k_minus) +
              A_center)
    
    # Store result
    tl.store(B + idx, result, mask=mask)

@triton.jit
def heat_3d_phase2_kernel(A, B, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles a block of elements from the 3D space
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear index to 3D coordinates (i, j, k)
    # For interior points: i,j,k in [1, N-2]
    interior_size = N - 2
    total_interior = interior_size * interior_size * interior_size
    
    mask = offsets < total_interior
    
    # Convert linear offset to 3D coordinates
    temp = offsets
    k = temp % interior_size + 1  # k in [1, N-2]
    temp = temp // interior_size
    j = temp % interior_size + 1  # j in [1, N-2]
    i = temp // interior_size + 1  # i in [1, N-2]
    
    # Calculate 3D array indices
    idx = i * N * N + j * N + k
    
    # Load central values
    B_center = tl.load(B + idx, mask=mask)
    
    # Load neighbor values for stencil computation
    B_i_plus = tl.load(B + idx + N * N, mask=mask)   # B[i+1][j][k]
    B_i_minus = tl.load(B + idx - N * N, mask=mask)  # B[i-1][j][k]
    B_j_plus = tl.load(B + idx + N, mask=mask)       # B[i][j+1][k]
    B_j_minus = tl.load(B + idx - N, mask=mask)      # B[i][j-1][k]
    B_k_plus = tl.load(B + idx + 1, mask=mask)       # B[i][j][k+1]
    B_k_minus = tl.load(B + idx - 1, mask=mask)      # B[i][j][k-1]
    
    # Compute stencil: 0.125 * second derivative + original value
    result = (0.125 * (B_i_plus - 2.0 * B_center + B_i_minus) +
              0.125 * (B_j_plus - 2.0 * B_center + B_j_minus) +
              0.125 * (B_k_plus - 2.0 * B_center + B_k_minus) +
              B_center)
    
    # Store result
    tl.store(A + idx, result, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    # Calculate grid dimensions
    interior_size = N - 2
    total_interior = interior_size * interior_size * interior_size
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_interior, BLOCK_SIZE),)
    
    # Time stepping loop in host code
    for t in range(1, TSTEPS + 1):
        # Phase 1: A -> B
        heat_3d_phase1_kernel[grid](A, B, N, BLOCK_SIZE)
        
        # Phase 2: B -> A (kernel launch provides synchronization)
        heat_3d_phase2_kernel[grid](A, B, N, BLOCK_SIZE)