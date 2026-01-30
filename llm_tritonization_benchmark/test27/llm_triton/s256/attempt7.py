import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    j_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_offset = j_offset + 1  # Start from j=1
    mask = j_offset < len_2d
    
    # Load a[j-1] values
    a_prev_vals = tl.load(a_ptr + j_offset - 1, mask=mask)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new_vals = 1.0 - a_prev_vals
    
    # Store a[j]
    tl.store(a_ptr + j_offset, a_new_vals, mask=mask)
    
    # Load bb[j][i] and d[j]
    bb_offset = j_offset * len_2d + i_val
    bb_vals = tl.load(bb_ptr + bb_offset, mask=mask)
    d_vals = tl.load(d_ptr + j_offset, mask=mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new_vals + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offset = j_offset * len_2d + i_val
    tl.store(aa_ptr + aa_offset, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 256
    
    for i in range(len_2d):
        grid = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)  # -1 because j starts from 1
        
        s256_kernel[grid](
            a, aa, bb, d, i, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )