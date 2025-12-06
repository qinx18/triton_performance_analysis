import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, a_copy_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] from copy (scalar)
    a_prev = tl.load(a_copy_ptr + (i - 1))
    
    # Load aa[j][i] for all j
    aa_offsets = j_offsets * LEN_2D + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load bb[j][i] for all j
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Compute a[i] = aa[0][i] - a[i-1] (using first j=0 as in original C)
    aa_0_i = tl.load(aa_ptr + i)
    a_new = aa_0_i - a_prev
    
    # Store a[i] (only one thread should do this)
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i, a_new)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_new = a_new + bb_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + aa_offsets, aa_new, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Create read-only copy of a
    a_copy = a.clone()
    
    # Sequential loop over i, parallel over j
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](
            a, aa, bb, a_copy, i, LEN_2D, BLOCK_SIZE
        )