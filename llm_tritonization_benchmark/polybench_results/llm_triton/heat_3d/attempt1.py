import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for parallelizing over time steps
    t_id = tl.program_id(0)
    t = t_id + 1
    
    if t > TSTEPS:
        return
    
    # Process all spatial points in blocks
    total_points = (N - 2) * (N - 2) * (N - 2)
    
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, total_points, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < total_points
        
        # Convert linear index to 3D coordinates
        remaining = current_offsets
        i = remaining // ((N - 2) * (N - 2)) + 1
        remaining = remaining % ((N - 2) * (N - 2))
        j = remaining // (N - 2) + 1
        k = remaining % (N - 2) + 1
        
        # First phase: A -> B
        # Calculate 3D indices for A accesses
        center_idx = i * (N * N) + j * N + k
        i_plus_idx = (i + 1) * (N * N) + j * N + k
        i_minus_idx = (i - 1) * (N * N) + j * N + k
        j_plus_idx = i * (N * N) + (j + 1) * N + k
        j_minus_idx = i * (N * N) + (j - 1) * N + k
        k_plus_idx = i * (N * N) + j * N + (k + 1)
        k_minus_idx = i * (N * N) + j * N + (k - 1)
        
        # Load values from A
        A_center = tl.load(A_ptr + center_idx, mask=mask)
        A_i_plus = tl.load(A_ptr + i_plus_idx, mask=mask)
        A_i_minus = tl.load(A_ptr + i_minus_idx, mask=mask)
        A_j_plus = tl.load(A_ptr + j_plus_idx, mask=mask)
        A_j_minus = tl.load(A_ptr + j_minus_idx, mask=mask)
        A_k_plus = tl.load(A_ptr + k_plus_idx, mask=mask)
        A_k_minus = tl.load(A_ptr + k_minus_idx, mask=mask)
        
        # Compute B values
        B_val = (0.125 * (A_i_plus - 2.0 * A_center + A_i_minus) +
                 0.125 * (A_j_plus - 2.0 * A_center + A_j_minus) +
                 0.125 * (A_k_plus - 2.0 * A_center + A_k_minus) +
                 A_center)
        
        # Store to B
        tl.store(B_ptr + center_idx, B_val, mask=mask)
    
    # Synchronization point - need to ensure all B values are written
    # before proceeding to second phase
    
    # Process all spatial points in blocks for second phase
    for block_start in range(0, total_points, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < total_points
        
        # Convert linear index to 3D coordinates
        remaining = current_offsets
        i = remaining // ((N - 2) * (N - 2)) + 1
        remaining = remaining % ((N - 2) * (N - 2))
        j = remaining // (N - 2) + 1
        k = remaining % (N - 2) + 1
        
        # Second phase: B -> A
        # Calculate 3D indices for B accesses
        center_idx = i * (N * N) + j * N + k
        i_plus_idx = (i + 1) * (N * N) + j * N + k
        i_minus_idx = (i - 1) * (N * N) + j * N + k
        j_plus_idx = i * (N * N) + (j + 1) * N + k
        j_minus_idx = i * (N * N) + (j - 1) * N + k
        k_plus_idx = i * (N * N) + j * N + (k + 1)
        k_minus_idx = i * (N * N) + j * N + (k - 1)
        
        # Load values from B
        B_center = tl.load(B_ptr + center_idx, mask=mask)
        B_i_plus = tl.load(B_ptr + i_plus_idx, mask=mask)
        B_i_minus = tl.load(B_ptr + i_minus_idx, mask=mask)
        B_j_plus = tl.load(B_ptr + j_plus_idx, mask=mask)
        B_j_minus = tl.load(B_ptr + j_minus_idx, mask=mask)
        B_k_plus = tl.load(B_ptr + k_plus_idx, mask=mask)
        B_k_minus = tl.load(B_ptr + k_minus_idx, mask=mask)
        
        # Compute A values
        A_val = (0.125 * (B_i_plus - 2.0 * B_center + B_i_minus) +
                 0.125 * (B_j_plus - 2.0 * B_center + B_j_minus) +
                 0.125 * (B_k_plus - 2.0 * B_center + B_k_minus) +
                 B_center)
        
        # Store to A
        tl.store(A_ptr + center_idx, A_val, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    
    # Launch kernel for each time step sequentially
    for t in range(1, TSTEPS + 1):
        grid = (1,)
        heat_3d_kernel[grid](
            A, B, N, t, BLOCK_SIZE=BLOCK_SIZE
        )