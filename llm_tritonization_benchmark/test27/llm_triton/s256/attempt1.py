import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_idx < len_2d
    
    j_idx = tl.program_id(1)
    
    # Load a[j-1] and a[j]
    if j_idx > 0:
        a_j_minus_1 = tl.load(a_ptr + (j_idx - 1))
    else:
        a_j_minus_1 = 0.0
    
    # Compute a[j] = 1.0 - a[j-1]
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j]
    tl.store(a_ptr + j_idx, a_j)
    
    # Load bb[j][i] and d[j] for vectorized computation
    bb_offsets = j_idx * len_2d + i_idx
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j_idx)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j_idx * len_2d + i_idx
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 32
    
    # Grid: (num_blocks_i, j_range)
    num_blocks_i = triton.cdiv(len_2d, BLOCK_SIZE)
    j_range = len_2d - 1  # j starts from 1
    
    grid = (num_blocks_i, j_range)
    
    s256_kernel[grid](
        a, aa, bb, d,
        len_2d=len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )