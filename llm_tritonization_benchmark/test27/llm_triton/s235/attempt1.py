import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr,
    len_2d,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    # First compute: a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j
    for j in range(1, len_2d):
        # Load aa[j-1][i]
        aa_prev_ptrs = aa_ptr + (j - 1) * len_2d + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j * len_2d + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j * len_2d + i_idx
        tl.store(aa_ptrs, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )