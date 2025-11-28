import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for j dimension (parallelized)
    pid_j = tl.program_id(0)
    
    # Sequential loop over i dimension (1 to LEN_2D-1)
    for i in range(1, LEN_2D):
        # Calculate j offsets for this block
        j_start = pid_j * BLOCK_SIZE
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Load a[i-1] (scalar broadcast)
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Load aa[j][i] values
        aa_offsets = j_offsets * LEN_2D + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Store a[i] (only first valid thread stores to avoid race)
        if pid_j == 0:
            tl.store(a_ptr + i, a_new)
        
        # Load bb[j][i] values
        bb_offsets = j_offsets * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        # Use the computed a_new value
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array 'a'
    a_copy = a.clone()
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s257_kernel[grid](
        a, a_copy, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )