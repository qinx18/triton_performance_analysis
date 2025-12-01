import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for i in range(1, LEN_2D):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_block_start in range(1, LEN_2D, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            mask = (j_indices < LEN_2D) & (j_indices >= 1)
            
            # Load aa[j-1][i]
            aa_prev_ptrs = aa_ptr + (j_indices - 1) * LEN_2D + i
            aa_prev_vals = tl.load(aa_prev_ptrs, mask=mask, other=0.0)
            
            # Load cc[j][i]
            cc_ptrs = cc_ptr + j_indices * LEN_2D + i
            cc_vals = tl.load(cc_ptrs, mask=mask, other=0.0)
            
            # Compute and store aa[j][i]
            result_vals = aa_prev_vals + cc_vals
            aa_curr_ptrs = aa_ptr + j_indices * LEN_2D + i
            tl.store(aa_curr_ptrs, result_vals, mask=mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for i in range(1, LEN_2D):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_block_start in range(1, LEN_2D, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            mask = (j_indices < LEN_2D) & (j_indices >= 1)
            
            # Load bb[i-1][j]
            bb_prev_ptrs = bb_ptr + (i - 1) * LEN_2D + j_indices
            bb_prev_vals = tl.load(bb_prev_ptrs, mask=mask, other=0.0)
            
            # Load cc[i][j]
            cc_ptrs = cc_ptr + i * LEN_2D + j_indices
            cc_vals = tl.load(cc_ptrs, mask=mask, other=0.0)
            
            # Compute and store bb[i][j]
            result_vals = bb_prev_vals + cc_vals
            bb_curr_ptrs = bb_ptr + i * LEN_2D + j_indices
            tl.store(bb_curr_ptrs, result_vals, mask=mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    grid = (1,)
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )