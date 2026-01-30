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
        valid_mask = j_mask & (j_idx >= i)
        
        # Load aa[j][i-1]
        prev_idx = j_idx * len_2d + (i - 1)
        aa_prev = tl.load(aa_ptr + prev_idx, mask=valid_mask, other=0.0)
        
        # Load bb[j][i]
        bb_idx = j_idx * len_2d + i
        bb_val = tl.load(bb_ptr + bb_idx, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        aa_new = aa_prev * aa_prev + bb_val
        
        # Store aa[j][i]
        curr_idx = j_idx * len_2d + i
        tl.store(aa_ptr + curr_idx, aa_new, mask=valid_mask)

def s232_triton(aa, bb, len_2d):
    BLOCK_SIZE = 64
    j_size = len_2d - 1  # j ranges from 1 to len_2d-1
    grid = (triton.cdiv(j_size, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, len_2d, BLOCK_SIZE=BLOCK_SIZE)