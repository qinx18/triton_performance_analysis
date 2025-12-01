import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    bb_ptr,
    d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_current = j_start + j_offsets
        j_mask = (j_current < LEN_2D) & (j_current >= 1)
        
        # Load a[j-1] from copy
        a_prev_offsets = j_current - 1
        a_prev_vals = tl.load(a_copy_ptr + a_prev_offsets, mask=j_mask)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_vals = 1.0 - a_prev_vals
        
        # Store a[j] to original array
        tl.store(a_ptr + j_current, a_vals, mask=j_mask)
        
        # Load d[j] and bb[j][i]
        d_vals = tl.load(d_ptr + j_current, mask=j_mask)
        bb_offsets = j_current * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_vals + bb_vals * d_vals
        
        # Store aa[j][i]
        aa_offsets = j_current * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        d,
        LEN_2D,
        BLOCK_SIZE,
    )