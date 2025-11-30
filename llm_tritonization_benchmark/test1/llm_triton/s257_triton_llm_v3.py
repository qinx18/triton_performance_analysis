import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension (starting from 1)
    i_block = tl.program_id(0)
    i = i_block + 1
    
    if i >= LEN_2D:
        return
    
    # Process all j values for this i
    j_start = 0
    j_remaining = LEN_2D
    
    while j_remaining > 0:
        # Calculate block size for this iteration
        current_block_size = min(BLOCK_SIZE, j_remaining)
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < (j_start + current_block_size)
        
        # Calculate offsets for 2D arrays (row-major: aa[j][i] = aa[j * LEN_2D + i])
        aa_offsets = j_offsets * LEN_2D + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        bb_vals = tl.load(bb_ptr + aa_offsets, mask=j_mask)
        a_prev = tl.load(a_copy_ptr + (i - 1))  # a[i-1] from read-only copy
        
        # Compute: a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Compute: aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store results
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)
        
        # Store a[i] (same value for all j, but we store it multiple times - last one wins)
        if j_mask:
            tl.store(a_ptr + i, a_new)
        
        j_start += current_block_size
        j_remaining -= current_block_size

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    # Grid size: process i from 1 to LEN_2D-1
    grid = (LEN_2D - 1,)
    
    # Block size for j dimension processing
    BLOCK_SIZE = 256
    
    s257_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )