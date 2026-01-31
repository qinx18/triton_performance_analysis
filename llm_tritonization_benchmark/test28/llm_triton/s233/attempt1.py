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
        # Load aa[j-1][i] and cc[j][i]
        aa_prev_idx = (j - 1) * len_2d + i_actual
        aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=i_mask, other=0.0)
        
        cc_idx = j * len_2d + i_actual
        cc_val = tl.load(cc_ptr + cc_idx, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + cc[j][i]
        result = aa_prev + cc_val
        
        # Store aa[j][i]
        aa_curr_idx = j * len_2d + i_actual
        tl.store(aa_ptr + aa_curr_idx, result, mask=i_mask)

@triton.jit
def s233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d - 1
    j_actual = j_idx + 1
    
    for i in range(1, len_2d):
        # Load bb[j][i-1] and cc[j][i]
        bb_prev_idx = j_actual * len_2d + (i - 1)
        bb_prev = tl.load(bb_ptr + bb_prev_idx, mask=j_mask, other=0.0)
        
        cc_idx = j_actual * len_2d + i
        cc_val = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        # Compute bb[j][i] = bb[j][i-1] + cc[j][i]
        result = bb_prev + cc_val
        
        # Store bb[j][i]
        bb_curr_idx = j_actual * len_2d + i
        tl.store(bb_ptr + bb_curr_idx, result, mask=j_mask)

def s233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    # First kernel: parallelize i, sequential j
    i_size = len_2d - 1
    grid_i = (triton.cdiv(i_size, BLOCK_SIZE),)
    s233_kernel_aa[grid_i](aa, cc, len_2d, BLOCK_SIZE)
    
    # Second kernel: parallelize j, sequential i
    j_size = len_2d - 1
    grid_j = (triton.cdiv(j_size, BLOCK_SIZE),)
    s233_kernel_bb[grid_j](bb, cc, len_2d, BLOCK_SIZE)