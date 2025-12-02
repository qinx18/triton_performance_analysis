import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, LEN_2D, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1  # Start from i=1
    
    mask = indices < LEN_2D
    
    # Load b[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load c[1] (scalar)
    c_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1]
    aa_k_indices = k * LEN_2D + (indices - 1)  # aa[k][i-1]
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_val
    
    # Store to aa[j][i]
    aa_j_indices = j * LEN_2D + indices
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[1]
    
    # We process from i=1 to LEN_2D-1, so total elements = LEN_2D-1
    n_elements = LEN_2D - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c,
        j, k,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa