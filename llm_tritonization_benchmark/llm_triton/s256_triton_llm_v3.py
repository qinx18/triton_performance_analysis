import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process j from 1 to LEN_2D-1 sequentially
    for j in range(1, LEN_2D):
        # Load a[j-1] from copy (read-only)
        a_prev = tl.load(a_copy_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new = 1.0 - a_prev
        
        # Store a[j] to original array
        tl.store(a_ptr + j, a_new)
        
        # Load bb[j][i] and d[j]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_val = a_new + bb_val * d_val
        
        # Store aa[j][i]
        tl.store(aa_ptr + j * LEN_2D + i, aa_val)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    # Launch kernel with one thread block per column i
    grid = (LEN_2D,)
    BLOCK_SIZE = 256
    
    s256_kernel[grid](
        a, a_copy, aa, bb, d,
        LEN_2D, BLOCK_SIZE
    )