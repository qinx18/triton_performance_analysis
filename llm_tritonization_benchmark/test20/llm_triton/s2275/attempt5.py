import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 N, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, 
                 stride_cc_0, stride_cc_1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # 2D array operations: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_ptrs = aa_ptr + j_idx * stride_aa_0 + i * stride_aa_1
        bb_ptrs = bb_ptr + j_idx * stride_bb_0 + i * stride_bb_1
        cc_ptrs = cc_ptr + j_idx * stride_cc_0 + i * stride_cc_1
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptrs, result, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i] (only thread 0 does this)
        if (pid == 0) & (tl.sum(j_offsets) == 0):
            a_ptr_i = a_ptr + i
            b_ptr_i = b_ptr + i
            c_ptr_i = c_ptr + i
            d_ptr_i = d_ptr + i
            
            b_val = tl.load(b_ptr_i)
            c_val = tl.load(c_ptr_i)
            d_val = tl.load(d_ptr_i)
            
            a_result = b_val + c_val * d_val
            tl.store(a_ptr_i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        N, aa.stride(0), aa.stride(1), bb.stride(0), bb.stride(1),
        cc.stride(0), cc.stride(1), BLOCK_SIZE
    )