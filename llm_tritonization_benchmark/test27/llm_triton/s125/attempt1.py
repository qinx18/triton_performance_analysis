import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start + offsets
    
    mask = k_offsets < len_2d * len_2d
    
    i_coords = k_offsets // len_2d
    j_coords = k_offsets % len_2d
    
    aa_indices = i_coords * len_2d + j_coords
    bb_indices = i_coords * len_2d + j_coords
    cc_indices = i_coords * len_2d + j_coords
    
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + k_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array, len_2d):
    total_elements = len_2d * len_2d
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )