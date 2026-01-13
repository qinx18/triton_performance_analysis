import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar broadcast)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # For each j, compute a[i] = aa[j][i] - a[i-1]
    # Since each j overwrites a[i], we need to process sequentially within the kernel
    for j in range(LEN_2D):
        # Load aa[j][i]
        aa_offset = j * LEN_2D + i
        aa_val = tl.load(aa_ptr + aa_offset)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_val = aa_val - a_prev
        
        # Store a[i]
        tl.store(a_ptr + i, a_val)
        
        # Load bb[j][i]
        bb_offset = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Compute and store aa[j][i] = a[i] + bb[j][i]
        new_aa_val = a_val + bb_val
        tl.store(aa_ptr + aa_offset, new_aa_val)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        BLOCK_SIZE = 1
        
        s257_kernel[(1,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )