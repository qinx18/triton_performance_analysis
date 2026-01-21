import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, 
                n_i, n_j, 
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n_i
    
    # Load and update a[i] = a[i] + b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
    
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j starting from 1
    for j in range(1, n_j):
        # Load aa[j-1][i] (previous row)
        prev_offsets = (j - 1) * n_i + i_idx
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=i_mask, other=0.0)
        
        # Load bb[j][i] (current row)
        bb_offsets = j * n_i + i_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new = aa_prev + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_offsets = j * n_i + i_idx
        tl.store(aa_ptr + aa_offsets, aa_new, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    n_j = aa.shape[0]
    n_i = aa.shape[1]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_i, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        n_i, n_j,
        BLOCK_SIZE=BLOCK_SIZE
    )