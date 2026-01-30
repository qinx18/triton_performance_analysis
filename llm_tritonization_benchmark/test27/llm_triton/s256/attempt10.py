import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j_val, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    i_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i_offset < len_2d
    
    # Load a[j] and a[j-1] (scalars for this j)
    a_prev = tl.load(a_ptr + j_val - 1)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j] (only need to do this once per j, but all threads will write same value)
    tl.store(a_ptr + j_val, a_new)
    
    # Load bb[j][i] and d[j] 
    bb_offset = j_val * len_2d + i_offset
    bb_vals = tl.load(bb_ptr + bb_offset, mask=mask)
    d_val = tl.load(d_ptr + j_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new + bb_vals * d_val
    
    # Store aa[j][i]
    aa_offset = j_val * len_2d + i_offset
    tl.store(aa_ptr + aa_offset, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 256
    
    for j in range(1, len_2d):
        grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
        
        s256_kernel[grid](
            a, aa, bb, d, j, len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )