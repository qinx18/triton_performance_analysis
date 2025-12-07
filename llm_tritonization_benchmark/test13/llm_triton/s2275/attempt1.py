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
        # aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_indices = j_idx * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + aa_indices, mask=j_mask)
        cc_vals = tl.load(cc_ptr + aa_indices, mask=j_mask)
        
        aa_new = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_indices, aa_new, mask=j_mask)
        
        # a[i] = b[i] + c[i] * d[i] (only for first thread)
        if pid == 0 and j_offsets[0] == 0:
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_new = b_val + c_val * d_val
            tl.store(a_ptr + i, a_new)

def s2275_triton(a, aa, b, bb, c, cc, d):
    len_2d = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )