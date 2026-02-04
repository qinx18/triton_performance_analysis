import triton
import triton.language as tl

@triton.jit
def s2233_kernel_aa(aa_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # Load aa[j-1][i] and cc[j][i]
        aa_prev_ptrs = aa_ptr + (j - 1) * len_2d + i_idx
        cc_ptrs = cc_ptr + j * len_2d + i_idx
        
        aa_prev = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_val = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + cc[j][i]
        result = aa_prev + cc_val
        
        # Store aa[j][i]
        aa_curr_ptrs = aa_ptr + j * len_2d + i_idx
        tl.store(aa_curr_ptrs, result, mask=i_mask)

@triton.jit
def s2233_kernel_bb(bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        # Load bb[i-1][j] and cc[i][j]
        bb_prev_ptrs = bb_ptr + (i - 1) * len_2d + j_idx
        cc_ptrs = cc_ptr + i * len_2d + j_idx
        
        bb_prev = tl.load(bb_prev_ptrs, mask=j_mask, other=0.0)
        cc_val = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        # Compute bb[i][j] = bb[i-1][j] + cc[i][j]
        result = bb_prev + cc_val
        
        # Store bb[i][j]
        bb_curr_ptrs = bb_ptr + i * len_2d + j_idx
        tl.store(bb_curr_ptrs, result, mask=j_mask)

def s2233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    
    # First kernel: update aa array, start from i=1
    grid1 = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s2233_kernel_aa[grid1](aa[:, 1:], cc, len_2d - 1, BLOCK_SIZE)
    
    # Second kernel: update bb array, start from j=1
    grid2 = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    s2233_kernel_bb[grid2](bb[1:, :], cc, len_2d - 1, BLOCK_SIZE)