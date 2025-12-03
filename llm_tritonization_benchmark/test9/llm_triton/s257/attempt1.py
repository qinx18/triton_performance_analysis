import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension (starting from 1)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process all j values for this i
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load a[i-1] from the read-only copy (scalar broadcast)
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Calculate 2D indices for aa[j][i] and bb[j][i]
        aa_indices = j_indices * LEN_2D + i
        bb_indices = j_indices * LEN_2D + i
        
        # Load aa[j][i] and bb[j][i]
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Store a[i] to original array (all j iterations write same value)
        if j_start == 0:  # Only store once per i
            tl.store(a_ptr + i, a_new[0])
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_indices, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array a to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 32
    grid = (LEN_2D - 1,)  # i ranges from 1 to LEN_2D-1
    
    s257_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE
    )