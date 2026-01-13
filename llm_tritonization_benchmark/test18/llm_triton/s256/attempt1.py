import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D, j_val, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    i_mask = i_offsets < LEN_2D
    
    # Load data for current j
    a_j = tl.load(a_ptr + j_val)
    a_j_minus_1 = tl.load(a_ptr + (j_val - 1))
    
    # Compute new a[j] value
    new_a_j = 1.0 - a_j_minus_1
    
    # Store updated a[j]
    tl.store(a_ptr + j_val, new_a_j)
    
    # Load bb[j][i] and d[j] for aa computation
    bb_offsets = j_val * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = new_a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j_val * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for i dimension
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        s256_kernel[(grid_size,)](
            a, aa, bb, d, 
            LEN_2D, j, BLOCK_SIZE
        )