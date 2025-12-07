import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = 1 + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Load from read-only copy for a[j-1]
    a_prev_vals = tl.load(a_copy_ptr + j_idx - 1, mask=j_mask)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j] values
    tl.store(a_ptr + j_idx, a_vals, mask=j_mask)
    
    # Load bb[j][i] and d[j] for aa computation
    bb_offsets = j_idx * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    d_vals = tl.load(d_ptr + j_idx, mask=j_mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i] values
    aa_offsets = j_idx * LEN_2D + i_val
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    # Sequential j iterations with parallel i processing
    for i in range(LEN_2D):
        # Launch kernel for current i value
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s256_kernel[grid](
            a, a_copy, aa, bb, d,
            i, LEN_2D, BLOCK_SIZE
        )
        
        # Update the read-only copy after each j-sequential step
        a_copy.copy_(a)