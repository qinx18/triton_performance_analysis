import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, 
                len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < len_2d
    
    # First computation: a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=mask_i)
    c_vals = tl.load(c_ptr + i_idx, mask=mask_i)
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)
    
    # Sequential loop for j dimension (dependencies require ordering)
    for j in range(1, len_2d):
        # Load aa[j-1][i] (previous row)
        aa_prev_offsets = (j - 1) * len_2d + i_idx
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=mask_i)
        
        # Load bb[j][i] (current row)
        bb_curr_offsets = j * len_2d + i_idx
        bb_curr_vals = tl.load(bb_ptr + bb_curr_offsets, mask=mask_i)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_curr_vals = aa_prev_vals + bb_curr_vals * a_vals
        
        # Store aa[j][i]
        aa_curr_offsets = j * len_2d + i_idx
        tl.store(aa_ptr + aa_curr_offsets, aa_curr_vals, mask=mask_i)

def s235_triton(a, aa, b, bb, c, len_2d):
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        len_2d=len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )