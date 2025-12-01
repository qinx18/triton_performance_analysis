import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential computation for i dimension
    for i in range(1, LEN_2D):
        # First inner loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        for j_block_start in range(1, LEN_2D, BLOCK_SIZE):
            j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < LEN_2D
            
            # Load aa[j-1][i]
            aa_prev_idx = (j_offsets - 1) * LEN_2D + i
            aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=j_mask)
            
            # Load cc[j][i]
            cc_idx = j_offsets * LEN_2D + i
            cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask)
            
            # Compute and store aa[j][i]
            aa_new = aa_prev + cc_vals
            aa_idx = j_offsets * LEN_2D + i
            tl.store(aa_ptr + aa_idx, aa_new, mask=j_mask)
        
        # Second inner loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        for j_block_start in range(1, LEN_2D, BLOCK_SIZE):
            j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < LEN_2D
            
            # Load bb[i-1][j]
            bb_prev_idx = (i - 1) * LEN_2D + j_offsets
            bb_prev = tl.load(bb_ptr + bb_prev_idx, mask=j_mask)
            
            # Load cc[i][j]
            cc_idx = i * LEN_2D + j_offsets
            cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask)
            
            # Compute and store bb[i][j]
            bb_new = bb_prev + cc_vals
            bb_idx = i * LEN_2D + j_offsets
            tl.store(bb_ptr + bb_idx, bb_new, mask=j_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (1,)
    s2233_kernel[grid](aa, bb, cc, LEN_2D, BLOCK_SIZE)