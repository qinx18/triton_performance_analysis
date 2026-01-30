import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d: tl.constexpr, j: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_idx < len_2d
    
    # Load a[j-1] and compute a[j] = 1.0 - a[j-1]
    a_j_minus_1 = tl.load(a_ptr + (j - 1))
    a_j = 1.0 - a_j_minus_1
    
    # Store a[j]
    tl.store(a_ptr + j, a_j)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j * len_2d + i_idx
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j * len_2d + i_idx
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = 256
    
    # Loop over j sequentially from 1 to len_2d-1
    for j in range(1, len_2d):
        num_blocks_i = triton.cdiv(len_2d, BLOCK_SIZE)
        grid = (num_blocks_i,)
        
        s256_kernel[grid](
            a, aa, bb, d,
            len_2d=len_2d,
            j=j,
            BLOCK_SIZE=BLOCK_SIZE
        )