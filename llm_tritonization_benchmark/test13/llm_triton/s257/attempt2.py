import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value)
    a_prev = tl.load(a_copy_ptr + (i_val - 1))
    
    # Load aa[0][i] to compute a[i] (only need first j for this computation)
    aa_0i = tl.load(aa_ptr + i_val)  # aa[0][i]
    
    # Compute a[i] = aa[0][i] - a[i-1] (same for all j)
    a_new = aa_0i - a_prev
    
    # Store a[i] (only first block writes this)
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i_val, a_new)
    
    # Load aa[j][i] values for all j in this block
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j in this block
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    # Use the computed a[i] value
    aa_new = a_new + bb_vals
    
    # Store aa[j][i] values
    tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of a
    a_copy = a.clone()
    
    # Block size for j dimension
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        # Calculate grid size for j dimension
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        
        # Launch kernel for this i value
        s257_kernel[grid](
            a, a_copy, aa, bb, i, LEN_2D, BLOCK_SIZE
        )