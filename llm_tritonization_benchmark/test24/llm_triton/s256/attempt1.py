import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Load bb[j][i] and d[j] values for all i
    bb_ptrs = bb_ptr + i_offsets * stride_bb_1
    bb_vals = tl.load(bb_ptrs, mask=i_mask)
    d_val = tl.load(d_ptr)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_prev_val = tl.load(a_ptr - 1)
    a_val = 1.0 - a_prev_val
    
    # Store a[j]
    tl.store(a_ptr, a_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] for all i
    aa_ptrs = aa_ptr + i_offsets * stride_aa_1
    tl.store(aa_ptrs, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[1]
    
    # Get strides for 2D arrays
    stride_aa_0, stride_aa_1 = aa.stride()
    stride_bb_0, stride_bb_1 = bb.stride()
    
    BLOCK_SIZE = 256
    
    # Sequential loop over j
    for j in range(1, N):
        # Calculate pointers for current j
        a_ptr = a + j
        aa_ptr = aa[j]
        bb_ptr = bb[j]
        d_ptr = d + j
        
        # Launch kernel for all i in parallel
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_kernel[grid](
            a_ptr, aa_ptr, bb_ptr, d_ptr, N,
            stride_aa_0, stride_aa_1,
            stride_bb_0, stride_bb_1,
            BLOCK_SIZE=BLOCK_SIZE
        )