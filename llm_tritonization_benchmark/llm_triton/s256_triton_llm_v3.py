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
    # Get the current block
    block_id = tl.program_id(0)
    i = block_id
    
    if i >= LEN_2D:
        return
    
    # Process j from 1 to LEN_2D-1 sequentially
    for j in range(1, LEN_2D):
        # Load a[j-1] from the read-only copy
        a_prev = tl.load(a_copy_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_val = 1.0 - a_prev
        
        # Store a[j] to original array
        tl.store(a_ptr + j, a_val)
        
        # Load bb[j][i] and d[j]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_val = a_val + bb_val * d_val
        
        # Store aa[j][i]
        tl.store(aa_ptr + j * LEN_2D + i, aa_val)

def s256_triton(a, aa, bb, d):
    LEN_2D = a.shape[0]
    
    # Create read-only copy of array a to handle WAR dependencies
    a_copy = a.clone()
    
    # Launch kernel with one block per i dimension
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        d,
        LEN_2D,
        BLOCK_SIZE=256,
    )