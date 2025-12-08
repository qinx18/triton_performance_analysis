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
    # Sequential processing of i dimension (strict requirement due to dependency)
    for i in range(1, LEN_2D):
        # Parallel processing of j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        # Load a[i-1] once (scalar value)
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Process j dimension in blocks
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = current_j < LEN_2D
            
            # Load aa[j][i] values for current block
            aa_offsets = current_j * LEN_2D + i
            aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            # Since a[i] should be same for all j, use first valid j
            if j_start == 0:
                a_new = tl.load(aa_ptr + i) - a_prev
                tl.store(a_ptr + i, a_new)
            
            # Load the computed a[i]
            a_current = tl.load(a_ptr + i)
            
            # Load bb[j][i] values
            bb_offsets = current_j * LEN_2D + i
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
            
            # Compute and store aa[j][i] = a[i] + bb[j][i]
            aa_new = a_current + bb_vals
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