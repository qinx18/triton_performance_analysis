import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask = i_idx < N
    
    # Load and update a[i] = a[i] + b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + i_idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=mask, other=0.0)
    new_a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, new_a_vals, mask=mask)
    
    # Sequential loop over j from 1 to N-1
    for j in range(1, N):
        # Load aa[j-1][i]
        prev_aa_offsets = (j-1) * N + i_idx
        prev_aa_vals = tl.load(aa_ptr + prev_aa_offsets, mask=mask, other=0.0)
        
        # Load bb[j][i]
        bb_offsets = j * N + i_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        new_aa_vals = prev_aa_vals + bb_vals * new_a_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + bb_offsets, new_aa_vals, mask=mask)

def s235_triton(a, aa, b, bb, c):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s235_kernel[grid](a, aa, b, bb, c, N=N, BLOCK_SIZE=BLOCK_SIZE)