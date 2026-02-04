import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel_aa(aa_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d - 1
    i_actual = i_idx + 1
    
    for j in range(1, len_2d):
        aa_curr_ptr = aa_ptr + j * len_2d + i_actual
        aa_prev_ptr = aa_ptr + (j - 1) * len_2d + i_actual
        cc_ptr_curr = cc_ptr + j * len_2d + i_actual
        
        aa_prev = tl.load(aa_prev_ptr, mask=i_mask)
        cc_val = tl.load(cc_ptr_curr, mask=i_mask)
        result = aa_prev + cc_val
        
        tl.store(aa_curr_ptr, result, mask=i_mask)

@triton.jit
def s233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d - 1
    j_actual = j_idx + 1
    
    for i in range(1, len_2d):
        bb_curr_ptr = bb_ptr + j_actual * len_2d + i
        bb_prev_ptr = bb_ptr + j_actual * len_2d + (i - 1)
        cc_ptr_curr = cc_ptr + j_actual * len_2d + i
        
        bb_prev = tl.load(bb_prev_ptr, mask=j_mask)
        cc_val = tl.load(cc_ptr_curr, mask=j_mask)
        result = bb_prev + cc_val
        
        tl.store(bb_curr_ptr, result, mask=j_mask)

def s233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    # First kernel: parallelize across i, sequential over j
    i_grid = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    s233_kernel_aa[(i_grid,)](aa, cc, len_2d, BLOCK_SIZE)
    
    # Second kernel: parallelize across j, sequential over i
    j_grid = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    s233_kernel_bb[(j_grid,)](bb, cc, len_2d, BLOCK_SIZE)