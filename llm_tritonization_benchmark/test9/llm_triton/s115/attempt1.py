import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, LEN_2D, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Only process i values where i > j
    mask = (i_offsets < LEN_2D) & (i_offsets > j)
    
    # Load a[i] values
    a_i = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_ji_offsets = j * LEN_2D + i_offsets
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=mask, other=0.0)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_i - aa_ji * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(LEN_2D):
        # Calculate number of i values to process (i = j+1 to LEN_2D-1)
        num_i = LEN_2D - (j + 1)
        if num_i <= 0:
            continue
            
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, LEN_2D, BLOCK_SIZE)