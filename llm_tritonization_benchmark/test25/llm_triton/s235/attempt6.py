import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once at the start
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    # First update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask, other=0.0)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask, other=0.0)
    
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension (data dependency)
    for j in range(1, len_2d):
        # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        prev_j_offsets = (j - 1) * len_2d + i_idx
        curr_j_offsets = j * len_2d + i_idx
        bb_offsets = j * len_2d + i_idx
        
        aa_prev_vals = tl.load(aa_ptr + prev_j_offsets, mask=i_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        tl.store(aa_ptr + curr_j_offsets, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )