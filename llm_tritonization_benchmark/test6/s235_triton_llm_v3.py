import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(
    a_ptr, b_ptr, c_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the outer i dimension
    pid = tl.program_id(axis=0)
    
    # Calculate the i index this program handles
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Update a[i] += b[i] * c[i]
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    a_val = a_val + b_val * c_val
    tl.store(a_ptr + i, a_val)
    
    # Sequential inner loop: aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
    # This has a dependency between iterations, so must be sequential
    for j in range(1, LEN_2D):
        # Load aa[j-1][i]
        aa_prev = tl.load(aa_ptr + (j-1) * LEN_2D + i)
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new = aa_prev + bb_val * a_val
        # Store aa[j][i]
        tl.store(aa_ptr + j * LEN_2D + i, aa_new)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (LEN_2D,)
    
    s235_kernel[grid](
        a, b, c, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )