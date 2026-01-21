import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_idx, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    j_offsets = offsets + 1  # j starts from 1
    mask = j_offsets < N
    
    # Load a[j-1]
    prev_offsets = offsets  # j-1 for j starting from 1
    prev_mask = j_offsets < N
    a_prev_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
    
    # Calculate a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_vals, mask=mask)
    
    # Load bb[j][i] and d[j]
    bb_vals = tl.load(bb_ptr + j_offsets * N + i_idx, mask=mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=mask)
    
    # Calculate aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + j_offsets * N + i_idx, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(N):
        # Launch kernel for j loop (j from 1 to N-1)
        grid = (triton.cdiv(N-1, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d, i, N, BLOCK_SIZE=BLOCK_SIZE
        )