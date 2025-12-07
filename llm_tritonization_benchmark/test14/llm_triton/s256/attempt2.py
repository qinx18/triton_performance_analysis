import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, j_idx, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] from the read-only copy
    a_j_minus_1 = tl.load(a_copy_ptr + (j_idx - 1))
    
    # Compute a[j] = 1.0 - a[j-1]
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j] to original array
    tl.store(a_ptr + j_idx, a_j)
    
    # Load d[j]
    d_j = tl.load(d_ptr + j_idx)
    
    # Load bb[j][i] for all valid i values
    bb_offsets = j_idx * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i] for all valid i values
    aa_offsets = j_idx * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = 256
    BLOCK_SIZE = 256
    
    # Sequential loop over j dimension (j starts from 1)
    for j in range(1, LEN_2D):
        # Create read-only copy of array a before each kernel launch
        a_copy = a.clone()
        
        # Launch kernel for current j, parallelizing over all i values
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s256_kernel[grid](
            a, a_copy, aa, bb, d,
            j, LEN_2D, BLOCK_SIZE
        )