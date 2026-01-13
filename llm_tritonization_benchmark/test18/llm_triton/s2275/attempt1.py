import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # 2D array operation: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_row_ptr = aa_ptr + i * len_2d
        bb_row_ptr = bb_ptr + i * len_2d
        cc_row_ptr = cc_ptr + i * len_2d
        
        aa_vals = tl.load(aa_row_ptr + j_idx, mask=j_mask)
        bb_vals = tl.load(bb_row_ptr + j_idx, mask=j_mask)
        cc_vals = tl.load(cc_row_ptr + j_idx, mask=j_mask)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_row_ptr + j_idx, result, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i] (only for first thread block)
        if pid == 0 and i < len_2d:
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_result = b_val + c_val * d_val
            tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    len_2d = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        len_2d=len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )