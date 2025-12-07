import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] from the read-only copy (scalar)
    a_i_minus_1 = tl.load(a_copy_ptr + (i_val - 1))
    
    # Load aa[j][i] values for all j
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j  
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # For j=0: a[i] = aa[0][i] - a[i-1]
    # Get the first aa value
    first_aa_val = tl.load(aa_ptr + i_val)  # aa[0][i]
    a_i_val = first_aa_val - a_i_minus_1
    
    # Store a[i] using the first thread only
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i_val, a_i_val)
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_vals = a_i_val + bb_vals
    
    # Store the new aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

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