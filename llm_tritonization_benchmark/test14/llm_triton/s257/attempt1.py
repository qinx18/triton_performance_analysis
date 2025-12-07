import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] from the read-only copy
    a_i_minus_1 = tl.load(a_copy_ptr + (i_val - 1))
    
    # Load aa[j][i] values for all j
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] (same value for all j)
    a_i_val = aa_vals - a_i_minus_1
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    new_aa_vals = a_i_val + bb_vals
    
    # Store the new aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)
    
    # Store a[i] (use first computed value since they're all the same)
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i_val, tl.load(aa_ptr + i_val))

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of a
    a_copy = a.clone()
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    grid = (1,)
    
    # Process each i sequentially
    for i in range(1, LEN_2D):
        s257_kernel[grid](
            a, a_copy, aa, bb, i, LEN_2D, BLOCK_SIZE
        )