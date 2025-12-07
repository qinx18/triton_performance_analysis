import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, n_i, n_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n_i
    
    # Load b[i] and c[i] once
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j
    for j in range(1, n_j):
        # Load aa[j-1][i]
        prev_aa_idx = (j - 1) * n_i + i_idx
        prev_aa_vals = tl.load(aa_ptr + prev_aa_idx, mask=i_mask, other=0.0)
        
        # Load bb[j][i]
        bb_idx = j * n_i + i_idx
        bb_vals = tl.load(bb_ptr + bb_idx, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_vals = prev_aa_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_idx = j * n_i + i_idx
        tl.store(aa_ptr + aa_idx, aa_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    n_i = 256
    n_j = 256
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_i, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa.view(-1), b, bb.view(-1), c,
        n_i, n_j,
        BLOCK_SIZE=BLOCK_SIZE
    )