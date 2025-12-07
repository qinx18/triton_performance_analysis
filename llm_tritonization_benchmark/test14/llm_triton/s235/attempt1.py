import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < LEN_2D
    
    # Load b and c for this block of i values
    b_vals = tl.load(b_ptr + i_idx, mask=mask_i)
    c_vals = tl.load(c_ptr + i_idx, mask=mask_i)
    
    # Load and update a[i] = a[i] + b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] for all i in this block
        aa_prev_ptr = aa_ptr + (j - 1) * LEN_2D + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=mask_i)
        
        # Load bb[j][i] for all i in this block
        bb_ptr_offset = bb_ptr + j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr_offset, mask=mask_i)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i] for all i in this block
        aa_curr_ptr = aa_ptr + j * LEN_2D + i_idx
        tl.store(aa_curr_ptr, aa_new_vals, mask=mask_i)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = 256
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )