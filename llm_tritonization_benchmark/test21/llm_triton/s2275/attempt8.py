import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr, 
                 N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # 2D computation: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_idx = i * N + j_idx
        bb_idx = i * N + j_idx
        cc_idx = i * N + j_idx
        
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)
        
        # 1D computation: a[i] = b[i] + c[i] * d[i] (only first thread in first block)
        if pid == 0:
            first_thread_mask = j_offsets == 0
            if tl.sum(first_thread_mask) > 0:
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                result_1d = b_val + c_val * d_val
                tl.store(a_ptr + i, result_1d)

def s2275_triton(a, aa, b, bb, c, cc, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2275_kernel[grid](aa, bb, cc, a, b, c, d, N, BLOCK_SIZE)