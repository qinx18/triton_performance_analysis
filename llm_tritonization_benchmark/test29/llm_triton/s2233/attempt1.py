import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel_aa(aa_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    valid_i = (i_idx >= 1) & (i_idx < len_2d)
    
    for j in range(1, len_2d):
        aa_curr_ptr = aa_ptr + j * len_2d + i_idx
        aa_prev_ptr = aa_ptr + (j - 1) * len_2d + i_idx
        cc_ptr_curr = cc_ptr + j * len_2d + i_idx
        
        aa_prev = tl.load(aa_prev_ptr, mask=i_mask & valid_i, other=0.0)
        cc_curr = tl.load(cc_ptr_curr, mask=i_mask & valid_i, other=0.0)
        
        result = aa_prev + cc_curr
        tl.store(aa_curr_ptr, result, mask=i_mask & valid_i)

@triton.jit
def s2233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    valid_j = (j_idx >= 1) & (j_idx < len_2d)
    
    for i in range(1, len_2d):
        bb_curr_ptr = bb_ptr + i * len_2d + j_idx
        bb_prev_ptr = bb_ptr + (i - 1) * len_2d + j_idx
        cc_ptr_curr = cc_ptr + i * len_2d + j_idx
        
        bb_prev = tl.load(bb_prev_ptr, mask=j_mask & valid_j, other=0.0)
        cc_curr = tl.load(cc_ptr_curr, mask=j_mask & valid_j, other=0.0)
        
        result = bb_prev + cc_curr
        tl.store(bb_curr_ptr, result, mask=j_mask & valid_j)

def s2233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    
    grid_i = (triton.cdiv(len_2d, BLOCK_SIZE),)
    s2233_kernel_aa[grid_i](aa, cc, len_2d, BLOCK_SIZE)
    
    grid_j = (triton.cdiv(len_2d, BLOCK_SIZE),)
    s2233_kernel_bb[grid_j](bb, cc, len_2d, BLOCK_SIZE)