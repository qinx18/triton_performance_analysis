import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j: tl.constexpr, k: tl.constexpr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1  # i starts from 1
    
    mask = i_offsets < LEN_2D
    
    # Load b[i] - use i_offsets directly since b is indexed by i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] - single scalar value
    c1_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] - k-th row, (i-1)-th column
    i_minus_1_offsets = i_offsets - 1
    aa_k_vals = tl.load(aa_ptr + k * LEN_2D + i_minus_1_offsets, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i] - j-th row, i-th column
    tl.store(aa_ptr + j * LEN_2D + i_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    n_elements = LEN_2D - 1  # Loop goes from i=1 to i<LEN_2D
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, LEN_2D, BLOCK_SIZE
    )