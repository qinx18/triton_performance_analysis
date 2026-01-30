import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    i_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i_offset < len_2d
    
    for j in range(1, len_2d):
        # Load a[j-1]
        a_val_prev = tl.load(a_ptr + j - 1)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_val_new = 1.0 - a_val_prev
        
        # Store a[j]
        tl.store(a_ptr + j, a_val_new)
        
        # Load bb[j][i] and d[j]
        bb_offset = j * len_2d + i_offset
        bb_vals = tl.load(bb_ptr + bb_offset, mask=mask)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_val_new + bb_vals * d_val
        
        # Store aa[j][i]
        aa_offset = j * len_2d + i_offset
        tl.store(aa_ptr + aa_offset, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s256_kernel[grid](
        a, aa, bb, d, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )