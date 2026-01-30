import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # Compute aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_idx = j_idx * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + aa_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + aa_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)
        
        # Only one thread handles a[i] = b[i] + c[i] * d[i] for each i
        if pid == 0 and j_offsets[0] == 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            tl.store(a_ptr + i, b_val + c_val * d_val)

def s2275_triton(a, aa, b, bb, c, cc, d, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d, len_2d, BLOCK_SIZE
    )