import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Define offsets once
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential processing of i dimension (strict requirement due to dependency)
    for i in range(1, LEN_2D):
        # Load a[i-1] (scalar value from read-only copy)
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Compute a[i] using aa[0][i] - a[i-1] (use first row as reference)
        aa_first = tl.load(aa_ptr + i)  # aa[0][i]
        a_new = aa_first - a_prev
        tl.store(a_ptr + i, a_new)
        
        # Process j dimension in parallel blocks
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = current_j < LEN_2D
            
            # Calculate 2D array offsets: aa[j][i] = j * LEN_2D + i
            aa_offsets = current_j * LEN_2D + i
            bb_offsets = current_j * LEN_2D + i
            
            # Load bb[j][i] values
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            aa_new = a_new + bb_vals
            tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy of array a to handle WAR dependency
    a_copy = a.clone()
    
    # Launch kernel with single thread block (sequential processing required)
    grid = (1,)
    s257_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE
    )