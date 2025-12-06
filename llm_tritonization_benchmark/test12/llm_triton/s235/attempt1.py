import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, n_elements, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = block_start + offsets
    mask = i_idx < n_elements
    
    # Load a, b, c values once
    a_vals = tl.load(a_ptr + i_idx, mask=mask)
    b_vals = tl.load(b_ptr + i_idx, mask=mask)
    c_vals = tl.load(c_ptr + i_idx, mask=mask)
    
    # Update a[i] = a[i] + b[i] * c[i]
    new_a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, new_a_vals, mask=mask)
    
    # Sequential loop over j
    for j in range(1, LEN_2D):
        # Load aa[j-1][i]
        prev_aa_idx = (j - 1) * LEN_2D + i_idx
        prev_aa_vals = tl.load(aa_ptr + prev_aa_idx, mask=mask)
        
        # Load bb[j][i]
        bb_idx = j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr + bb_idx, mask=mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_vals = prev_aa_vals + bb_vals * new_a_vals
        
        # Store aa[j][i]
        aa_idx = j * LEN_2D + i_idx
        tl.store(aa_ptr + aa_idx, aa_vals, mask=mask)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = aa.shape[0]
    n_elements = LEN_2D
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        n_elements, LEN_2D, BLOCK_SIZE
    )