import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i < LEN_2D
    
    # First inner loop: for j in range(LEN_2D)
    for j in range(LEN_2D):
        # Calculate linear indices for aa[j][i], bb[j][i], cc[j][i]
        idx = j * LEN_2D + i
        
        aa_val = tl.load(aa_ptr + idx, mask=i_mask)
        bb_val = tl.load(bb_ptr + idx, mask=i_mask)
        cc_val = tl.load(cc_ptr + idx, mask=i_mask)
        
        result = aa_val + bb_val * cc_val
        tl.store(aa_ptr + idx, result, mask=i_mask)
    
    # Second computation: a[i] = b[i] + c[i] * d[i]
    a_val = tl.load(a_ptr + i, mask=i_mask)
    b_val = tl.load(b_ptr + i, mask=i_mask)
    c_val = tl.load(c_ptr + i, mask=i_mask)
    d_val = tl.load(d_ptr + i, mask=i_mask)
    
    result = b_val + c_val * d_val
    tl.store(a_ptr + i, result, mask=i_mask)

def s2275_triton(aa, bb, cc, a, b, c, d):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )