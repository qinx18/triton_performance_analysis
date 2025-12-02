import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    k_indices = block_start + offsets
    
    mask = k_indices < LEN_2D * LEN_2D
    
    i_indices = k_indices // LEN_2D
    j_indices = k_indices % LEN_2D
    
    aa_offsets = i_indices * LEN_2D + j_indices
    bb_offsets = i_indices * LEN_2D + j_indices
    cc_offsets = i_indices * LEN_2D + j_indices
    
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + k_indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array, 
        LEN_2D, BLOCK_SIZE
    )