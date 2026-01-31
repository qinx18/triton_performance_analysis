import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < len_2d * len_2d
    
    # Load data using offsets
    aa_vals = tl.load(aa_ptr + indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + indices, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array, len_2d):
    # Get total number of elements
    total_elements = len_2d * len_2d
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel
    s125_kernel[(num_blocks,)](
        aa, bb, cc, flat_2d_array,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )