import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # 2D array updates: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_idx = j_idx * LEN_2D + i
        bb_idx = j_idx * LEN_2D + i  
        cc_idx = j_idx * LEN_2D + i
        
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)
        
        # 1D array update: a[i] = b[i] + c[i] * d[i]
        # Only update once per i (when pid == 0 and j_idx[0] < LEN_2D)
        if pid == 0:
            if i < LEN_2D:
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                result_1d = b_val + c_val * d_val
                tl.store(a_ptr + i, result_1d)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )