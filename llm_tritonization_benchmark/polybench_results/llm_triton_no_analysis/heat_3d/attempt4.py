import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 3D block
    pid = tl.program_id(0)
    
    # Calculate total interior elements
    interior_size = N - 2
    total_interior = interior_size * interior_size * interior_size
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for vectorized access
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Mask for valid elements
    mask = element_ids < total_interior
    
    # Convert linear index to 3D coordinates (i, j, k)
    interior_size_sq = interior_size * interior_size
    
    i_interior = element_ids // interior_size_sq
    remainder = element_ids % interior_size_sq
    j_interior = remainder // interior_size
    k_interior = remainder % interior_size
    
    # Convert to actual array indices (add 1 since interior starts at index 1)
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1
    
    # Calculate linear indices for 3D array access
    N_sq = N * N
    center_idx = i * N_sq + j * N + k
    
    # Precompute neighbor offsets
    ip1_offset = N_sq      # i+1
    im1_offset = -N_sq     # i-1
    jp1_offset = N         # j+1
    jm1_offset = -N        # j-1
    kp1_offset = 1         # k+1
    km1_offset = -1        # k-1
    
    # Time stepping loop
    for t in range(1, TSTEPS + 1):
        # First phase: A -> B
        # Load A values with stencil pattern
        a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        a_ip1 = tl.load(A_ptr + center_idx + ip1_offset, mask=mask, other=0.0)
        a_im1 = tl.load(A_ptr + center_idx + im1_offset, mask=mask, other=0.0)
        a_jp1 = tl.load(A_ptr + center_idx + jp1_offset, mask=mask, other=0.0)
        a_jm1 = tl.load(A_ptr + center_idx + jm1_offset, mask=mask, other=0.0)
        a_kp1 = tl.load(A_ptr + center_idx + kp1_offset, mask=mask, other=0.0)
        a_km1 = tl.load(A_ptr + center_idx + km1_offset, mask=mask, other=0.0)
        
        # Compute B[i][j][k]
        b_val = (0.125 * (a_ip1 - 2.0 * a_center + a_im1) +
                 0.125 * (a_jp1 - 2.0 * a_center + a_jm1) +
                 0.125 * (a_kp1 - 2.0 * a_center + a_km1) +
                 a_center)
        
        # Store B values
        tl.store(B_ptr + center_idx, b_val, mask=mask)
        
        # Second phase: B -> A
        # Load B values with stencil pattern
        b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        b_ip1 = tl.load(B_ptr + center_idx + ip1_offset, mask=mask, other=0.0)
        b_im1 = tl.load(B_ptr + center_idx + im1_offset, mask=mask, other=0.0)
        b_jp1 = tl.load(B_ptr + center_idx + jp1_offset, mask=mask, other=0.0)
        b_jm1 = tl.load(B_ptr + center_idx + jm1_offset, mask=mask, other=0.0)
        b_kp1 = tl.load(B_ptr + center_idx + kp1_offset, mask=mask, other=0.0)
        b_km1 = tl.load(B_ptr + center_idx + km1_offset, mask=mask, other=0.0)
        
        # Compute A[i][j][k]
        a_val = (0.125 * (b_ip1 - 2.0 * b_center + b_im1) +
                 0.125 * (b_jp1 - 2.0 * b_center + b_jm1) +
                 0.125 * (b_kp1 - 2.0 * b_center + b_km1) +
                 b_center)
        
        # Store A values
        tl.store(A_ptr + center_idx, a_val, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    # Calculate interior elements
    interior_size = N - 2
    total_interior = interior_size * interior_size * interior_size
    
    # Block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(total_interior, BLOCK_SIZE)
    
    # Launch kernel
    heat_3d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )