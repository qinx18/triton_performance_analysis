import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(
    aa_ptr,
    aa_copy_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get current position in the flattened 2D array
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert flat indices to 2D coordinates
    j_coords = offsets // LEN_2D
    i_coords = offsets % LEN_2D
    
    # Create mask for valid elements (j >= 1, i >= 1, within bounds)
    mask = (offsets < LEN_2D * LEN_2D) & (j_coords >= 1) & (i_coords >= 1)
    
    # Calculate offsets for aa[j][i-1] and aa[j-1][i]
    left_offsets = offsets - 1  # aa[j][i-1]
    up_offsets = offsets - LEN_2D  # aa[j-1][i]
    
    # Load values from the read-only copy
    left_vals = tl.load(aa_copy_ptr + left_offsets, mask=mask)
    up_vals = tl.load(aa_copy_ptr + up_offsets, mask=mask)
    
    # Compute result
    result = (left_vals + up_vals) / 1.9
    
    # Store to original array
    tl.store(aa_ptr + offsets, result, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Configure kernel launch
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s2111_kernel[grid](
        aa,
        aa_copy,
        LEN_2D,
        BLOCK_SIZE,
    )