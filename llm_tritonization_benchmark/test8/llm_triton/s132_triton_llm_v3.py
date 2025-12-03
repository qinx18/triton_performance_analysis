import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, LEN_2D, j, k):
    BLOCK_SIZE = 64
    pid = tl.program_id(axis=0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = pid * BLOCK_SIZE + offsets + 1
    
    mask = i_offsets < LEN_2D
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] - single value
    c_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values
    aa_k_indices = k * LEN_2D + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute result: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_val
    
    # Store to aa[j][i]
    aa_j_indices = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[1]
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s132_kernel[grid](aa, b, c, LEN_2D, j, k)
    
    return aa