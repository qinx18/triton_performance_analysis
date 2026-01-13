import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask = i_idx < LEN_2D
    
    # First compute a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + i_idx, mask=mask)
    c_vals = tl.load(c_ptr + i_idx, mask=mask)
    a_vals = tl.load(a_ptr + i_idx, mask=mask)
    
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_idx, a_vals, mask=mask)
    
    # Sequential loop over j
    for j in range(1, LEN_2D):
        # Load aa[j-1][i]
        prev_aa_vals = tl.load(aa_ptr + (j - 1) * LEN_2D + i_idx, mask=mask)
        
        # Load bb[j][i]
        bb_vals = tl.load(bb_ptr + j * LEN_2D + i_idx, mask=mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_vals = prev_aa_vals + bb_vals * a_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + j * LEN_2D + i_idx, aa_vals, mask=mask)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )