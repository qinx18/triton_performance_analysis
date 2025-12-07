import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Process each i sequentially
    for i in range(1, LEN_2D):
        # Load a[i-1] from the read-only copy (scalar)
        a_i_minus_1 = tl.load(a_copy_ptr + (i - 1))
        
        # First, compute a[i] = aa[0][i] - a[i-1] (using j=0)
        aa_0_i_offset = i  # aa[0][i] in row-major layout
        aa_0_i = tl.load(aa_ptr + aa_0_i_offset)
        a_i = aa_0_i - a_i_minus_1
        
        # Store a[i]
        tl.store(a_ptr + i, a_i)
        
        # Load bb[j][i] values for all j
        bb_offsets = j_offsets * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i] for all j
        aa_offsets = j_offsets * LEN_2D + i
        new_aa_vals = a_i + bb_vals
        
        # Store the new aa[j][i] values
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of a
    a_copy = a.clone()
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    grid = (1,)
    
    s257_kernel[grid](
        a, a_copy, aa, bb, LEN_2D, BLOCK_SIZE
    )