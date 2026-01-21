import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n
    
    # Load b and c values for all i in this block
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    
    # Update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j
    for j in range(1, n):
        # Load aa[j-1][i] values
        aa_prev_offsets = (j - 1) * n + i_idx
        aa_prev_mask = i_mask
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=aa_prev_mask)
        
        # Load bb[j][i] values
        bb_offsets = j * n + i_idx
        bb_mask = i_mask
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=bb_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i] values
        aa_offsets = j * n + i_idx
        aa_mask = i_mask
        tl.store(aa_ptr + aa_offsets, aa_new_vals, mask=aa_mask)

def s235_triton(a, aa, b, bb, c):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s235_kernel[grid](a, aa, b, bb, c, n, BLOCK_SIZE=BLOCK_SIZE)