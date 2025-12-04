import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # Calculate 2D indices for aa, bb, cc arrays
        aa_indices = j_idx * LEN_2D + i
        
        # Load data for 2D arrays
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + aa_indices, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + aa_indices, mask=j_mask, other=0.0)
        
        # Compute and store 2D array update
        aa_new = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_indices, aa_new, mask=j_mask)
        
        # Update 1D arrays (only first thread in each i iteration)
        if pid == 0 and j_idx[0] == 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_new = b_val + c_val * d_val
            tl.store(a_ptr + i, a_new)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa.flatten(), b, bb.flatten(), c, cc.flatten(), d,
        LEN_2D, BLOCK_SIZE
    )