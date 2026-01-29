import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel_aa(aa_ptr, cc_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d - 1
    i_actual = i_idx + 1
    
    for j in range(1, len_2d):
        aa_curr_ptrs = aa_ptr + j * len_2d + i_actual
        aa_prev_ptrs = aa_ptr + (j - 1) * len_2d + i_actual
        cc_ptrs = cc_ptr + j * len_2d + i_actual
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask)
        cc_vals = tl.load(cc_ptrs, mask=i_mask)
        result = aa_prev_vals + cc_vals
        tl.store(aa_curr_ptrs, result, mask=i_mask)

@triton.jit
def s2233_kernel_bb(bb_ptr, cc_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d - 1
    j_actual = j_idx + 1
    
    for i in range(1, len_2d):
        bb_curr_ptrs = bb_ptr + i * len_2d + j_actual
        bb_prev_ptrs = bb_ptr + (i - 1) * len_2d + j_actual
        cc_ptrs = cc_ptr + i * len_2d + j_actual
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=j_mask)
        cc_vals = tl.load(cc_ptrs, mask=j_mask)
        result = bb_prev_vals + cc_vals
        tl.store(bb_curr_ptrs, result, mask=j_mask)

def s2233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    grid_aa = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s2233_kernel_aa[grid_aa](
        aa, cc, len_2d, BLOCK_SIZE
    )
    
    grid_bb = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s2233_kernel_bb[grid_bb](
        bb, cc, len_2d, BLOCK_SIZE
    )