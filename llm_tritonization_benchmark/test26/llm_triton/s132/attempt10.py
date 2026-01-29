import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j: tl.constexpr, k: tl.constexpr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1  # i starts from 1
    
    mask = i_offsets < LEN_2D
    
    # Load b[i] - b is 1D array indexed by i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] - single scalar value from 1D c array
    c1_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] - 2D array access: row k, column (i-1)
    i_minus_1 = i_offsets - 1
    aa_k_indices = k * LEN_2D + i_minus_1
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute result: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i] - 2D array access: row j, column i
    aa_j_indices = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    n_elements = LEN_2D - 1  # Loop from i=1 to i<LEN_2D
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, LEN_2D, BLOCK_SIZE
    )