import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Vectorized i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + 1  # Start from i=1
    i_mask = i_indices < LEN_2D
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        # Load aa[j-1][i] for all i values
        aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_indices
        aa_prev = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load cc[j][i] for all i values
        cc_ptrs = cc_ptr + j * LEN_2D + i_indices
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Compute and store aa[j][i]
        aa_new = aa_prev + cc_vals
        aa_ptrs = aa_ptr + j * LEN_2D + i_indices
        tl.store(aa_ptrs, aa_new, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        # Load bb[j][i-1] for all i values
        bb_prev_ptrs = bb_ptr + j * LEN_2D + (i_indices - 1)
        bb_prev = tl.load(bb_prev_ptrs, mask=i_mask, other=0.0)
        
        # Compute and store bb[j][i] (cc_vals already loaded)
        bb_new = bb_prev + cc_vals
        bb_ptrs = bb_ptr + j * LEN_2D + i_indices
        tl.store(bb_ptrs, bb_new, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Grid parallelizes over i dimension (excluding i=0)
    i_size = LEN_2D - 1  # i ranges from 1 to LEN_2D-1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )