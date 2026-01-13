import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, 
                n_i, n_j,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n_i
    
    # Load and update a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, n_j):
        # Load aa[j-1][i] for each i in block
        aa_prev_ptrs = aa_ptr + (j-1) * n_i + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask)
        
        # Load bb[j][i] for each i in block
        bb_ptrs = bb_ptr + j * n_i + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j * n_i + i_idx
        tl.store(aa_ptrs, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    n_i = aa.shape[1]
    n_j = aa.shape[0]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_i, BLOCK_SIZE),)
    
    s235_kernel[grid](a, aa, b, bb, c,
                      n_i, n_j,
                      BLOCK_SIZE=BLOCK_SIZE)