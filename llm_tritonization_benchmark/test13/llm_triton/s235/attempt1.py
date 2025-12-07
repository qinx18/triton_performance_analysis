import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=i_mask)
    c_vals = tl.load(c_ptr + i_idx, mask=i_mask)
    
    # Load and update a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] - previous row
        prev_idx = (j - 1) * LEN_2D + i_idx
        prev_vals = tl.load(aa_ptr + prev_idx, mask=i_mask)
        
        # Load bb[j][i] - current row
        curr_bb_idx = j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr + curr_bb_idx, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        curr_idx = j * LEN_2D + i_idx
        new_vals = prev_vals + bb_vals * a_vals
        tl.store(aa_ptr + curr_idx, new_vals, mask=i_mask)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )