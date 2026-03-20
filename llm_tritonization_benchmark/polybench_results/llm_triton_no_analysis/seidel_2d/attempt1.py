import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which (i,j) this program handles
    total_elements = (N - 2) * (N - 2)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    mask = element_ids < total_elements
    
    # Convert linear element_ids to (i,j) coordinates
    # element_id = (i-1) * (N-2) + (j-1), so i and j are in range [1, N-2]
    i_coords = (element_ids // (N - 2)) + 1
    j_coords = (element_ids % (N - 2)) + 1
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Load 9-point stencil values
        idx_center = i_coords * N + j_coords
        idx_top_left = (i_coords - 1) * N + (j_coords - 1)
        idx_top = (i_coords - 1) * N + j_coords
        idx_top_right = (i_coords - 1) * N + (j_coords + 1)
        idx_left = i_coords * N + (j_coords - 1)
        idx_right = i_coords * N + (j_coords + 1)
        idx_bottom_left = (i_coords + 1) * N + (j_coords - 1)
        idx_bottom = (i_coords + 1) * N + j_coords
        idx_bottom_right = (i_coords + 1) * N + (j_coords + 1)
        
        # Load all values
        val_top_left = tl.load(A_ptr + idx_top_left, mask=mask, other=0.0)
        val_top = tl.load(A_ptr + idx_top, mask=mask, other=0.0)
        val_top_right = tl.load(A_ptr + idx_top_right, mask=mask, other=0.0)
        val_left = tl.load(A_ptr + idx_left, mask=mask, other=0.0)
        val_center = tl.load(A_ptr + idx_center, mask=mask, other=0.0)
        val_right = tl.load(A_ptr + idx_right, mask=mask, other=0.0)
        val_bottom_left = tl.load(A_ptr + idx_bottom_left, mask=mask, other=0.0)
        val_bottom = tl.load(A_ptr + idx_bottom, mask=mask, other=0.0)
        val_bottom_right = tl.load(A_ptr + idx_bottom_right, mask=mask, other=0.0)
        
        # Compute average
        result = (val_top_left + val_top + val_top_right +
                 val_left + val_center + val_right +
                 val_bottom_left + val_bottom + val_bottom_right) / 9.0
        
        # Store result
        tl.store(A_ptr + idx_center, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    # Ensure A is contiguous
    A = A.contiguous()
    
    # Calculate number of elements to process (interior points only)
    total_elements = (N - 2) * (N - 2)
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )
    
    return A