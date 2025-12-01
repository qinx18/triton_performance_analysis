import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(
    a_ptr, b_ptr, c_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_block in range(0, LEN_2D, BLOCK_SIZE):
        current_i_offsets = i_block + i_offsets
        i_mask = current_i_offsets < LEN_2D
        
        # a[i] += b[i] * c[i]
        a_vals = tl.load(a_ptr + current_i_offsets, mask=i_mask)
        b_vals = tl.load(b_ptr + current_i_offsets, mask=i_mask)
        c_vals = tl.load(c_ptr + current_i_offsets, mask=i_mask)
        a_vals = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_i_offsets, a_vals, mask=i_mask)
        
        # Sequential loop over j
        for j in range(1, LEN_2D):
            # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
            aa_prev_vals = tl.load(aa_ptr + (j-1) * LEN_2D + current_i_offsets, mask=i_mask)
            bb_vals = tl.load(bb_ptr + j * LEN_2D + current_i_offsets, mask=i_mask)
            aa_new_vals = aa_prev_vals + bb_vals * a_vals
            tl.store(aa_ptr + j * LEN_2D + current_i_offsets, aa_new_vals, mask=i_mask)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s235_kernel[grid](
        a, b, c, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )