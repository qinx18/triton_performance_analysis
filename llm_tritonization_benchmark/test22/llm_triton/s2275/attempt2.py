import triton
import triton.language as tl

@triton.jit
def s2275_kernel(aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr, 
                 len_2d, len_1d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # 2D array operations: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_indices = i * len_2d + j_idx
        bb_indices = i * len_2d + j_idx
        cc_indices = i * len_2d + j_idx
        
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_indices, result, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i] (only first thread does this)
        if pid == 0:
            first_j = tl.load(j_offsets)
            if first_j == 0:
                if i < len_1d:
                    a_val = tl.load(a_ptr + i)
                    b_val = tl.load(b_ptr + i)
                    c_val = tl.load(c_ptr + i)
                    d_val = tl.load(d_ptr + i)
                    result_1d = b_val + c_val * d_val
                    tl.store(a_ptr + i, result_1d)

def s2275_triton(a, aa, b, bb, c, cc, d):
    len_2d = aa.shape[0]
    len_1d = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        len_2d, len_1d, BLOCK_SIZE=BLOCK_SIZE
    )