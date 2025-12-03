import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # First inner loop: for j in range(LEN_2D)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_offset = j_indices * LEN_2D + i
        bb_offset = j_indices * LEN_2D + i
        cc_offset = j_indices * LEN_2D + i
        
        aa_vals = tl.load(aa_ptr + aa_offset, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_offset, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_offset, mask=j_mask)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_offset, result, mask=j_mask)
    
    # Second statement: a[i] = b[i] + c[i] * d[i]
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    
    result_a = b_val + c_val * d_val
    tl.store(a_ptr + i, result_a)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )