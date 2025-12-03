import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        mask = (j_indices < LEN_2D) & (j_indices >= 1)
        
        # Load a[j-1] from copy
        a_prev_indices = j_indices - 1
        a_prev_mask = (a_prev_indices >= 0) & (a_prev_indices < LEN_2D) & mask
        a_prev_vals = tl.load(a_copy_ptr + a_prev_indices, mask=a_prev_mask, other=0.0)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_vals = 1.0 - a_prev_vals
        
        # Store a[j] to original array
        tl.store(a_ptr + j_indices, a_vals, mask=mask)
        
        # Load bb[j][i] and d[j]
        bb_indices = j_indices * LEN_2D + i
        bb_mask = (j_indices < LEN_2D) & (j_indices >= 1)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=bb_mask, other=0.0)
        d_vals = tl.load(d_ptr + j_indices, mask=bb_mask, other=0.0)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_vals + bb_vals * d_vals
        
        # Store aa[j][i]
        aa_indices = j_indices * LEN_2D + i
        tl.store(aa_ptr + aa_indices, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy of a to handle WAR dependencies
    a_copy = a.clone()
    
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a, a_copy, aa, bb, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )