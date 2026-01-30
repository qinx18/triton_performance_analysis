import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # 2D array indexing: aa[j][i] means j*len_2d + i
        aa_idx = j_idx * len_2d + i
        bb_idx = j_idx * len_2d + i
        cc_idx = j_idx * len_2d + i
        
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)
        
        # Update 1D arrays only when j_idx == 0 to avoid race conditions
        if pid == 0:
            first_j = tl.arange(0, 1)
            if tl.sum(first_j) == 0:  # Only first thread in first block
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                
                a_result = b_val + c_val * d_val
                tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d, len_2d):
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        len_2d, BLOCK_SIZE
    )