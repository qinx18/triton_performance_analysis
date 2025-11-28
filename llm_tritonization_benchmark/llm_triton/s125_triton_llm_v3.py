import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(flat_2d_array_ptr, aa_ptr, bb_ptr, cc_ptr, 
                len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (len_2d * len_2d)
    
    # Convert linear index to 2D indices
    i = offset // len_2d
    j = offset % len_2d
    
    # Calculate 2D array indices
    aa_idx = i * len_2d + j
    bb_idx = i * len_2d + j
    cc_idx = i * len_2d + j
    
    # Load values
    aa_val = tl.load(aa_ptr + aa_idx, mask=mask, other=0.0)
    bb_val = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
    cc_val = tl.load(cc_ptr + cc_idx, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_val + bb_val * cc_val
    
    # Store result
    tl.store(flat_2d_array_ptr + offset, result, mask=mask)

def s125_triton(flat_2d_array, aa, bb, cc):
    len_2d = aa.shape[0]
    total_elements = len_2d * len_2d
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        flat_2d_array, aa, bb, cc,
        len_2d=len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array