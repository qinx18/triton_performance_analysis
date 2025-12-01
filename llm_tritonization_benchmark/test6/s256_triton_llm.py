import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(
    a_ptr, aa_ptr, bb_ptr, d_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for the i dimension (outer loop)
    pid_i = tl.program_id(0)
    
    # Calculate the column index
    i = pid_i
    
    # Process sequentially through j dimension due to data dependency
    # a[j] depends on a[j-1], so we can't parallelize the j loop
    for j in range(1, LEN_2D):
        # Load a[j-1] to compute a[j] = 1.0 - a[j-1]
        a_prev = tl.load(a_ptr + j - 1)
        a_curr = 1.0 - a_prev
        
        # Store updated a[j]
        tl.store(a_ptr + j, a_curr)
        
        # Load bb[j, i] and d[j] for aa computation
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j, i] = a[j] + bb[j, i] * d[j]
        aa_val = a_curr + bb_val * d_val
        
        # Store aa[j, i]
        tl.store(aa_ptr + j * LEN_2D + i, aa_val)

def s256_triton(a, aa, bb, d):
    """
    Triton implementation of TSVC s256.
    
    Key optimizations:
    - Parallel execution across i dimension (columns)
    - Sequential processing of j dimension to handle data dependency
    - Direct memory access patterns for efficient GPU memory usage
    """
    a = a.contiguous()
    aa = aa.contiguous()
    bb = bb.contiguous()
    d = d.contiguous()
    
    LEN_2D = a.shape[0]
    
    # Launch one thread block per column (i dimension)
    grid = (LEN_2D,)
    
    # Use block size of 1 since we're processing one column per program
    BLOCK_SIZE = 1
    
    s256_kernel[grid](
        a, aa, bb, d,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, aa