import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel_aa(aa_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid_i * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # Load aa[j-1][i]
        aa_prev_ptrs = aa_ptr + (j-1) * len_2d + i_idx
        aa_prev = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptrs = cc_ptr + j * len_2d + i_idx
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + cc[j][i]
        result = aa_prev + cc_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j * len_2d + i_idx
        tl.store(aa_ptrs, result, mask=i_mask)

@triton.jit
def s233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid_j = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid_j * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        # Load bb[j][i-1]
        bb_prev_ptrs = bb_ptr + j_idx * len_2d + (i-1)
        bb_prev = tl.load(bb_prev_ptrs, mask=j_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptrs = cc_ptr + j_idx * len_2d + i
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        # Compute bb[j][i] = bb[j][i-1] + cc[j][i]
        result = bb_prev + cc_vals
        
        # Store bb[j][i]
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        tl.store(bb_ptrs, result, mask=j_mask)

def s233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    
    # First kernel: update aa array (parallelize over i)
    grid_i = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s233_kernel_aa[grid_i](
        aa, cc, len_2d, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: update bb array (parallelize over j)
    grid_j = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s233_kernel_bb[grid_j](
        bb, cc, len_2d, BLOCK_SIZE=BLOCK_SIZE
    )