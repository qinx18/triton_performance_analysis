import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get column index for this block
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Update a[i] = a[i] + b[i] * c[i]
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    new_a_val = a_val + b_val * c_val
    tl.store(a_ptr + i, new_a_val)
    
    # Sequential loop for dependency chain in column i
    # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i] for j in range(1, LEN_2D)
    prev_aa = tl.load(aa_ptr + 0 * LEN_2D + i)  # aa[0][i]
    
    for j in range(1, LEN_2D):
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        new_aa = prev_aa + bb_val * new_a_val
        tl.store(aa_ptr + j * LEN_2D + i, new_aa)
        prev_aa = new_aa

def s235_triton(a, aa, b, bb, c):
    """
    Triton optimized version of s235 - nested loops with dependency
    
    Key optimizations:
    - Parallelizes across columns (i dimension)
    - Keeps sequential dependency within each column
    - Coalesced memory access for row-major aa array
    """
    a = a.contiguous()
    aa = aa.contiguous()
    b = b.contiguous()
    bb = bb.contiguous()
    c = c.contiguous()
    
    LEN_2D = a.shape[0]
    
    # Launch kernel with one thread per column
    grid = (LEN_2D,)
    BLOCK_SIZE = 1
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, aa