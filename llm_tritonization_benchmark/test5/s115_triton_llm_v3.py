import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, LEN_2D: tl.constexpr, j_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Create offset vector once at the start
    base_offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = pid * BLOCK_SIZE + base_offsets
    
    # Mask for valid i indices that are greater than j_val
    i_mask = (i_offsets < LEN_2D) & (i_offsets > j_val)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j_val)
    
    # Load aa[j][i] values
    aa_ji_offsets = j_val * LEN_2D + i_offsets
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=i_mask, other=0.0)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s115_triton(a, aa):
    LEN_2D = aa.size(0)
    BLOCK_SIZE = 256
    
    for j in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s115_kernel[grid](
            a, aa, LEN_2D, j, BLOCK_SIZE
        )