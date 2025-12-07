import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    i_offset = pid * BLOCK_SIZE
    
    # Create offset vectors
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid i indices
    i_mask = i_offsets < LEN_2D
    
    # Only the first thread updates a[j]
    if pid == 0:
        # Load a[j-1]
        a_j_minus_1 = tl.load(a_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_j = 1.0 - a_j_minus_1
        
        # Store a[j]
        tl.store(a_ptr + j, a_j)
    
    # Load the updated a[j] value
    a_j = tl.load(a_ptr + j)
    
    # Load bb[j][i] and d[j] for the vectorized computation
    bb_offsets = j * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = 256
    BLOCK_SIZE = 64
    
    # Grid configuration - parallelize over i dimension
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        s256_kernel[grid](
            a, aa, bb, d, j,
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )