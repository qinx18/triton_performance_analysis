import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        i_mask = j_idx >= i
        valid_mask = j_mask & i_mask
        
        # Load aa[j][i-1]
        aa_prev_ptrs = aa_ptr + j_idx * len_2d + (i - 1)
        aa_prev = tl.load(aa_prev_ptrs, mask=valid_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = aa_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptr + j_idx * len_2d + i, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        tl.store(aa_ptrs, result, mask=valid_mask)

def s232_triton(aa, bb, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )