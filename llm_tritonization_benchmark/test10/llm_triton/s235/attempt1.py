import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    # Load b[i] and c[i] for this block
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    
    # Update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] for previous row
        aa_prev_offset = (j - 1) * LEN_2D + i_idx
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offset, mask=i_mask)
        
        # Load bb[j][i] for current row
        bb_offset = j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr + bb_offset, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        aa_offset = j * LEN_2D + i_idx
        tl.store(aa_ptr + aa_offset, aa_new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )