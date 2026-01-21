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
        # Vectorized 2D array operations for all j values in this block
        aa_ptrs = aa_ptr + j_idx * stride_aa_0 + i * stride_aa_1
        bb_ptrs = bb_ptr + j_idx * stride_bb_0 + i * stride_bb_1
        cc_ptrs = cc_ptr + j_idx * stride_cc_0 + i * stride_cc_1
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptrs, result, mask=j_mask)
        
        # 1D array operation - only first thread in block does this
        first_j = j_offsets == 0
        if pid == 0 and tl.sum(first_j.to(tl.int32)) > 0:
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
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        N, aa.stride(0), aa.stride(1), bb.stride(0), bb.stride(1),
        cc.stride(0), cc.stride(1), BLOCK_SIZE
    )