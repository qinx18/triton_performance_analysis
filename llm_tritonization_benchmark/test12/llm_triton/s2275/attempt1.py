import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # 2D array operations: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        bb_ptrs = bb_ptr + j_idx * LEN_2D + i
        cc_ptrs = cc_ptr + j_idx * LEN_2D + i
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        cc_vals = tl.load(cc_ptrs, mask=j_mask)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptrs, result, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i]
        # Only one thread per block should handle this to avoid race conditions
        if pid == 0 and j_offsets[0] == 0:
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_result = b_val + c_val * d_val
            tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )