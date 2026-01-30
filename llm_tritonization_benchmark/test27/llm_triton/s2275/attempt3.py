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
        aa_ptrs = aa_ptr + i * len_2d + j_idx
        bb_ptrs = bb_ptr + i * len_2d + j_idx
        cc_ptrs = cc_ptr + i * len_2d + j_idx
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptrs, result, mask=j_mask)
        
        # Compute a[i] = b[i] + c[i] * d[i] (only first thread in first block)
        first_thread = pid == 0
        first_element = j_idx < 1
        condition = first_thread & first_element
        if tl.sum(condition.to(tl.int32)) > 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            result_1d = b_val + c_val * d_val
            tl.store(a_ptr + i, result_1d)

def s2275_triton(a, aa, b, bb, c, cc, d, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d, len_2d, BLOCK_SIZE
    )