import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(len_2d):
        # Calculate valid i indices for this j (i >= j)
        i_start = j
        i_block_start = pid * BLOCK_SIZE
        
        # Current i indices for this block
        i_indices = i_block_start + i_offsets
        
        # Mask for valid elements: i >= j and i < len_2d
        mask = (i_indices >= i_start) & (i_indices < len_2d)
        
        if tl.sum(mask.to(tl.int32)) > 0:  # Only proceed if there are valid elements
            # Calculate linear indices for 2D arrays
            linear_idx = i_indices * len_2d + j
            
            # Load data
            bb_vals = tl.load(bb_ptr + linear_idx, mask=mask)
            cc_vals = tl.load(cc_ptr + linear_idx, mask=mask)
            
            # Compute
            result = bb_vals + cc_vals
            
            # Store result
            tl.store(aa_ptr + linear_idx, result, mask=mask)

def s1232_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    
    # Get dimensions
    n_elements = len_2d * len_2d
    
    # Calculate grid size for i dimension
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    # Launch kernel
    s1232_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )