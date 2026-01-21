import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    for j in range(1, LEN_2D):
        # Load a[j-1] (scalar load)
        a_prev = tl.load(a_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_j = 1.0 - a_prev
        
        # Store a[j] (scalar store)
        tl.store(a_ptr + j, a_j)
        
        # Load bb[j][i] for all valid i values
        bb_offsets = j * LEN_2D + i_offsets
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
        
        # Load d[j] (scalar load)
        d_j = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_j + bb_vals * d_j
        
        # Store aa[j][i] for all valid i values
        aa_offsets = j * LEN_2D + i_offsets
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s256_kernel[grid](
        a, aa, bb, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )