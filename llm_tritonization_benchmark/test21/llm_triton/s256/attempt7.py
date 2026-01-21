import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < LEN_2D
    
    for j in range(1, LEN_2D):
        # Load a[j-1]
        a_prev = tl.load(a_ptr + j - 1)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new = 1.0 - a_prev
        
        # Store a[j]
        tl.store(a_ptr + j, a_new)
        
        # Load bb[j][i] and d[j]
        bb_offsets = j * LEN_2D + i_offsets
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_new + bb_vals * d_val
        
        # Store aa[j][i]
        aa_offsets = j * LEN_2D + i_offsets
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    grid = (num_blocks,)
    s256_kernel[grid](a, aa, bb, d, LEN_2D, BLOCK_SIZE)