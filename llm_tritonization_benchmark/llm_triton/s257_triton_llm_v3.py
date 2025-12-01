import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension (starting from 1)
    i_idx = tl.program_id(0) + 1
    
    if i_idx >= LEN_2D:
        return
    
    # Process all j values for this i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_current = j_start + j_offsets
        j_mask = j_current < LEN_2D
        
        # Load a[i-1] (scalar, same for all j)
        a_prev = tl.load(a_copy_ptr + (i_idx - 1))
        
        # Load aa[j][i] values
        aa_offsets = j_current * LEN_2D + i_idx
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Load bb[j][i] values
        bb_offsets = j_current * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store a[i] (only need to store once per i, use first valid j)
        if j_start == 0:
            tl.store(a_ptr + i_idx, tl.sum(a_new * tl.where(j_mask, 1.0, 0.0)) / tl.sum(tl.where(j_mask, 1.0, 0.0)))
        
        # Store aa[j][i] values
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy for a
    a_copy = a.clone()
    
    # Launch kernel for i = 1 to LEN_2D-1
    BLOCK_SIZE = 64
    grid = (LEN_2D - 1,)
    
    s257_kernel[grid](
        a, a_copy, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )