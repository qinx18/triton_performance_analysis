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
    
    # Load aa[j][i] values for all j
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] values for all j
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1]
    # Since all j iterations compute the same value for a[i], we can compute it once
    a_new = aa_vals - a_prev
    
    # For a[i], we need to store the same value (use the first valid computation)
    # Since all j give same result, take the first one
    if tl.program_id(0) == 0:
        first_aa = tl.load(aa_ptr + i_val)  # aa[0][i]
        a_result = first_aa - a_prev
        tl.store(a_ptr + i_val, a_result)
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    # We need the new a[i] value
    a_new_scalar = tl.load(aa_ptr + i_val) - a_prev  # aa[0][i] - a[i-1]
    aa_new = a_new_scalar + bb_vals
    
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