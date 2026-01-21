import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, 
                N, i_val, BLOCK_SIZE: tl.constexpr):
    # Get thread block
    pid = tl.program_id(0)
    
    # Calculate j range for this block
    j_start = pid * BLOCK_SIZE + 1  # j starts from 1
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load a[j-1] values
    a_prev_offsets = j_offsets - 1
    a_prev_mask = (a_prev_offsets >= 0) & j_mask
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_new, mask=j_mask)
    
    # Load d[j] and bb[j][i]
    d_vals = tl.load(d_ptr + j_offsets, mask=j_mask, other=0.0)
    bb_offsets = j_offsets * stride_bb_0 + i_val * stride_bb_1
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_new = a_new + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * stride_aa_0 + i_val * stride_aa_1
    tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    
    for i in range(N):
        grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d,
            aa.stride(0), aa.stride(1),
            bb.stride(0), bb.stride(1),
            N, i, 
            BLOCK_SIZE=BLOCK_SIZE
        )