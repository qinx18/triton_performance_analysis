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
    
    # Process j from 1 to LEN_2D-1
    j_start = 1
    j_end = LEN_2D
    
    for j_block in range(j_start, j_end, BLOCK_SIZE):
        j_offsets = j_block + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets >= j_start) & (j_offsets < j_end)
        
        # Load a[j-1] from copy (read-only)
        a_prev_offsets = j_offsets - 1
        a_prev_mask = j_mask & (a_prev_offsets >= 0) & (a_prev_offsets < LEN_2D)
        a_prev_vals = tl.load(a_copy_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
        
        # Load d[j]
        d_vals = tl.load(d_ptr + j_offsets, mask=j_mask, other=0.0)
        
        # Load bb[j][i]
        bb_offsets = j_offsets * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new_vals = 1.0 - a_prev_vals
        
        # Store a[j] to original array
        tl.store(a_ptr + j_offsets, a_new_vals, mask=j_mask)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_new_vals + bb_vals * d_vals
        
        # Store aa[j][i]
        aa_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy of a to handle WAR dependency
    a_copy = a.clone()
    
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )