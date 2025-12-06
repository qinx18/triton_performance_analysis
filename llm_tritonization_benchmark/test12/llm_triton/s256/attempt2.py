import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, i, j, LEN_2D, BLOCK_SIZE_I: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] from read-only copy
    a_prev = tl.load(a_copy_ptr + (j - 1))
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j, a_new)
    
    # Load bb[j][i] and d[j] for vectorized computation
    bb_vals = tl.load(bb_ptr + j * LEN_2D + i_offsets, mask=i_mask)
    d_val = tl.load(d_ptr + j)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new + bb_vals * d_val
    
    # Store aa[j][i]
    tl.store(aa_ptr + j * LEN_2D + i_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE_I = 256
    
    for j in range(1, LEN_2D):
        # Create read-only copy for this iteration
        a_copy = a.clone()
        
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
        s256_kernel[grid](
            a, a_copy, aa, bb, d, 0, j, LEN_2D, BLOCK_SIZE_I
        )