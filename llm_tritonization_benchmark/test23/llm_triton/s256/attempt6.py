import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, j: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    
    # Calculate block offsets for i dimension
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] and a[j] (scalar values)
    a_j_minus_1 = tl.load(a_ptr + (j - 1))
    
    # Compute a[j] = 1.0 - a[j-1]
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j] (scalar store)
    tl.store(a_ptr + j, a_j)
    
    # Load d[j] (scalar)
    d_j = tl.load(d_ptr + j)
    
    # Load bb[j][i] values for all valid i in this block
    bb_offsets = j * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i] values
    aa_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Sequential loop over j dimension (dependency requires this)
    for j in range(1, LEN_2D):
        # Parallel execution over i dimension
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d,
            LEN_2D=LEN_2D,
            j=j,
            BLOCK_SIZE=BLOCK_SIZE
        )