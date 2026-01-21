import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load bb[j][i] and d[j]
    bb_vals = tl.load(bb_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load a[j-1]
    prev_offsets = offsets - 1
    prev_mask = mask & (offsets > 0)
    a_prev_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
    
    # Calculate a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j]
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Calculate aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(N):
        # Set up pointers for column i
        bb_i_ptr = bb[:, i]
        aa_i_ptr = aa[:, i]
        
        # Launch kernel for j loop (starting from j=1)
        grid = (triton.cdiv(N-1, BLOCK_SIZE),)
        s256_kernel[grid](
            a[1:], aa_i_ptr[1:], bb_i_ptr[1:], d[1:], 
            N-1, BLOCK_SIZE=BLOCK_SIZE
        )