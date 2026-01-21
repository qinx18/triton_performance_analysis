import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, 
                N, BLOCK_SIZE: tl.constexpr):
    # Get thread block for i dimension
    pid = tl.program_id(0)
    
    # Calculate i range for this block
    i_start = pid * BLOCK_SIZE
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Process j sequentially from 1 to N-1
    for j in range(1, N):
        # Load a[j-1] for all threads (broadcast)
        a_prev = tl.load(a_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new = 1.0 - a_prev
        
        # Store a[j] (all threads write same value, but that's ok)
        tl.store(a_ptr + j, a_new)
        
        # Load d[j] (broadcast to all threads)
        d_val = tl.load(d_ptr + j)
        
        # Load bb[j][i] for each thread's i value
        bb_offsets = j * stride_bb_0 + i_offsets * stride_bb_1
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_new = a_new + bb_vals * d_val
        
        # Store aa[j][i]
        aa_offsets = j * stride_aa_0 + i_offsets * stride_aa_1
        tl.store(aa_ptr + aa_offsets, aa_new, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s256_kernel[grid](
        a, aa, bb, d,
        aa.stride(0), aa.stride(1),
        bb.stride(0), bb.stride(1),
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )