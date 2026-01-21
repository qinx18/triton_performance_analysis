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

@triton.jit
def s2275_1d_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    a_ptrs = a_ptr + i_idx
    b_ptrs = b_ptr + i_idx
    c_ptrs = c_ptr + i_idx
    d_ptrs = d_ptr + i_idx
    
    b_vals = tl.load(b_ptrs, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptrs, mask=i_mask, other=0.0)
    d_vals = tl.load(d_ptrs, mask=i_mask, other=0.0)
    
    result = b_vals + c_vals * d_vals
    tl.store(a_ptrs, result, mask=i_mask)

def s2275_triton(a, aa, b, bb, c, cc, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch 2D array kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2275_kernel[grid](
        aa, aa, bb, bb, cc, cc, d,
        N, aa.stride(0), aa.stride(1), bb.stride(0), bb.stride(1),
        cc.stride(0), cc.stride(1), BLOCK_SIZE
    )
    
    # Launch 1D array kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2275_1d_kernel[grid](
        a, b, c, d, N, BLOCK_SIZE
    )