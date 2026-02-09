import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A, B, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 3D blocking
    pid = tl.program_id(0)
    
    # Calculate 3D coordinates from linear program ID
    total_elements = (N - 2) * (N - 2) * (N - 2)
    
    # Calculate base offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    element_offsets = block_start + offsets
    
    # Time stepping loop
    for t in range(TSTEPS):
        # First phase: A -> B
        # Convert linear indices to 3D coordinates (offset by 1 for interior points)
        valid_mask = element_offsets < total_elements
        
        k_coords = element_offsets % (N - 2) + 1
        temp_coords = element_offsets // (N - 2)
        j_coords = temp_coords % (N - 2) + 1
        i_coords = temp_coords // (N - 2) + 1
        
        # Calculate memory offsets for 3D array access
        center_idx = i_coords * N * N + j_coords * N + k_coords
        
        # Load center points and neighbors
        A_center = tl.load(A + center_idx, mask=valid_mask, other=0.0)
        
        # i-direction neighbors
        A_i_plus = tl.load(A + (i_coords + 1) * N * N + j_coords * N + k_coords, mask=valid_mask, other=0.0)
        A_i_minus = tl.load(A + (i_coords - 1) * N * N + j_coords * N + k_coords, mask=valid_mask, other=0.0)
        
        # j-direction neighbors
        A_j_plus = tl.load(A + i_coords * N * N + (j_coords + 1) * N + k_coords, mask=valid_mask, other=0.0)
        A_j_minus = tl.load(A + i_coords * N * N + (j_coords - 1) * N + k_coords, mask=valid_mask, other=0.0)
        
        # k-direction neighbors
        A_k_plus = tl.load(A + i_coords * N * N + j_coords * N + (k_coords + 1), mask=valid_mask, other=0.0)
        A_k_minus = tl.load(A + i_coords * N * N + j_coords * N + (k_coords - 1), mask=valid_mask, other=0.0)
        
        # Compute heat equation
        B_new = (0.125 * (A_i_plus - 2.0 * A_center + A_i_minus) +
                 0.125 * (A_j_plus - 2.0 * A_center + A_j_minus) +
                 0.125 * (A_k_plus - 2.0 * A_center + A_k_minus) +
                 A_center)
        
        # Store to B
        tl.store(B + center_idx, B_new, mask=valid_mask)
        
        # Second phase: B -> A
        # Load center points and neighbors from B
        B_center = tl.load(B + center_idx, mask=valid_mask, other=0.0)
        
        # i-direction neighbors
        B_i_plus = tl.load(B + (i_coords + 1) * N * N + j_coords * N + k_coords, mask=valid_mask, other=0.0)
        B_i_minus = tl.load(B + (i_coords - 1) * N * N + j_coords * N + k_coords, mask=valid_mask, other=0.0)
        
        # j-direction neighbors
        B_j_plus = tl.load(B + i_coords * N * N + (j_coords + 1) * N + k_coords, mask=valid_mask, other=0.0)
        B_j_minus = tl.load(B + i_coords * N * N + (j_coords - 1) * N + k_coords, mask=valid_mask, other=0.0)
        
        # k-direction neighbors
        B_k_plus = tl.load(B + i_coords * N * N + j_coords * N + (k_coords + 1), mask=valid_mask, other=0.0)
        B_k_minus = tl.load(B + i_coords * N * N + j_coords * N + (k_coords - 1), mask=valid_mask, other=0.0)
        
        # Compute heat equation
        A_new = (0.125 * (B_i_plus - 2.0 * B_center + B_i_minus) +
                 0.125 * (B_j_plus - 2.0 * B_center + B_j_minus) +
                 0.125 * (B_k_plus - 2.0 * B_center + B_k_minus) +
                 B_center)
        
        # Store to A
        tl.store(A + center_idx, A_new, mask=valid_mask)

def heat_3d_triton(A, B, N, TSTEPS):
    # Calculate interior elements
    interior_elements = (N - 2) * (N - 2) * (N - 2)
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(interior_elements, BLOCK_SIZE)
    
    # Launch kernel
    heat_3d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )