import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel_aa(aa_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
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
def s233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d - 1
    j_actual = j_idx + 1
    
    for i in range(1, len_2d):
        bb_curr_ptrs = bb_ptr + j_actual * len_2d + i
        bb_prev_ptrs = bb_ptr + j_actual * len_2d + (i - 1)
        cc_ptrs = cc_ptr + j_actual * len_2d + i
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=j_mask)
        cc_vals = tl.load(cc_ptrs, mask=j_mask)
        
        result = bb_prev_vals + cc_vals
        tl.store(bb_curr_ptrs, result, mask=j_mask)

def s233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    
    i_size = len_2d - 1
    grid_i = (triton.cdiv(i_size, BLOCK_SIZE),)
    s233_kernel_aa[grid_i](aa, cc, len_2d, BLOCK_SIZE)
    
    j_size = len_2d - 1
    grid_j = (triton.cdiv(j_size, BLOCK_SIZE),)
    s233_kernel_bb[grid_j](bb, cc, len_2d, BLOCK_SIZE)