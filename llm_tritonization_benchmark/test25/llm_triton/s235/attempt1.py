import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, n_i, n_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n_i
    
    # Load b and c vectors
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    
    # Load and update a values
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, n_j):
        # Load aa[j-1][i] values
        aa_prev_ptr = aa_ptr + (j - 1) * n_i + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=i_mask)
        
        # Load bb[j][i] values
        bb_ptr_curr = bb_ptr + j * n_i + i_idx
        bb_vals = tl.load(bb_ptr_curr, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i] values
        aa_curr_ptr = aa_ptr + j * n_i + i_idx
        tl.store(aa_curr_ptr, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    n_i = aa.shape[1]
    n_j = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_i, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        n_i, n_j,
        BLOCK_SIZE=BLOCK_SIZE
    )