import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr,
    len_2d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # 2D array operation: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_idx = j_idx * len_2d + i
        bb_idx = j_idx * len_2d + i
        cc_idx = j_idx * len_2d + i
        
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)
    
    # 1D array operation: a[i] = b[i] + c[i] * d[i] (only for first thread)
    if pid == 0:
        i_offsets = tl.arange(0, BLOCK_SIZE)
        for i_block in range(0, len_2d, BLOCK_SIZE):
            i_idx = i_block + i_offsets
            i_mask = i_idx < len_2d
            
            a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
            b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
            c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
            d_vals = tl.load(d_ptr + i_idx, mask=i_mask, other=0.0)
            
            result_1d = b_vals + c_vals * d_vals
            tl.store(a_ptr + i_idx, result_1d, mask=i_mask)

def s2275_triton(a, aa, b, bb, c, cc, d, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )