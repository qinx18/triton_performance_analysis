import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < len_2d * len_2d
    
    # Load values from 2D arrays using linear indexing
    aa_vals = tl.load(aa_ptr + current_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + current_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + current_offsets, mask=mask)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + current_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array, len_2d):
    # Calculate total number of elements
    n_elements = len_2d * len_2d
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s125_kernel[(grid_size,)](
        aa, bb, cc, flat_2d_array, len_2d, BLOCK_SIZE
    )