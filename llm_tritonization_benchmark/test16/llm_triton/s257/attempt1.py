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
    # Sequential processing of i dimension
    for i in range(1, LEN_2D):
        # Parallel processing of j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = current_j < LEN_2D
            
            # Load a[i-1] from read-only copy
            a_prev = tl.load(a_copy_ptr + (i - 1))
            
            # Load aa[j][i] values
            aa_offsets = current_j * LEN_2D + i
            aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            # Note: This is computed for each j, but result is same for all j
            a_new = aa_vals - a_prev
            
            # Load bb[j][i] values
            bb_offsets = current_j * LEN_2D + i
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            # Use the first valid a_new value for consistency
            if j_start == 0:
                # Store a[i] using first computed value
                tl.store(a_ptr + i, tl.sum(a_new) / tl.sum(j_mask.to(tl.float32)))
            
            # Load the stored a[i] value
            a_stored = tl.load(a_ptr + i)
            
            # Compute and store aa[j][i] = a[i] + bb[j][i]
            aa_new = a_stored + bb_vals
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