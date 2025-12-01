import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        mask = j_indices < LEN_2D
        
        # Load a[j-1] values from copy
        a_prev_indices = j_indices - 1
        a_prev_mask = (j_indices > 0) & (j_indices < LEN_2D)
        a_prev = tl.load(a_copy_ptr + a_prev_indices, mask=a_prev_mask, other=0.0)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new = 1.0 - a_prev
        
        # Store a[j] to original array
        tl.store(a_ptr + j_indices, a_new, mask=mask)
        
        # Load bb[j][i] and d[j]
        bb_indices = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + j_indices, mask=mask, other=0.0)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_new + bb_vals * d_vals
        
        # Store aa[j][i]
        aa_indices = j_indices * LEN_2D + i
        tl.store(aa_ptr + aa_indices, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a, a_copy, aa, bb, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )