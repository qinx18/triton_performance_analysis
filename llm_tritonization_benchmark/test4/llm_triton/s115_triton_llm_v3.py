import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base_offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = pid * BLOCK_SIZE + base_offsets
    
    # Only process i values where i > j_val
    valid_mask = (i_offsets < LEN_2D) & (i_offsets > j_val)
    
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + i_offsets, mask=valid_mask, other=0.0)
    
    # Load a[j_val] (scalar broadcast)
    a_j_val = tl.load(a_ptr + j_val)
    
    # Load aa[j_val][i] values
    aa_offsets = j_val * LEN_2D + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=valid_mask, other=0.0)
    
    # Compute: a[i] -= aa[j_val][i] * a[j_val]
    result = a_i_vals - aa_vals * a_j_val
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=valid_mask)

def s115_triton(a, aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s115_kernel[grid](
            a, aa, j, LEN_2D, BLOCK_SIZE
        )