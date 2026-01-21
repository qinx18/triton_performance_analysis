import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # We need indices from 1 to LEN_2D-1
    i_indices = offsets + 1
    mask = i_indices < LEN_2D
    
    # Load c[1]
    c1_val = tl.load(c_ptr + 1)
    
    # Load b[i] where i starts from 1
    b_vals = tl.load(b_ptr + i_indices, mask=mask)
    
    # Load aa[k][i-1] - convert 2D index to 1D
    k_row_offset = k * LEN_2D
    aa_k_vals = tl.load(aa_ptr + k_row_offset + (i_indices - 1), mask=mask)
    
    # Compute result: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i] - convert 2D index to 1D
    j_row_offset = j * LEN_2D
    tl.store(aa_ptr + j_row_offset + i_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Number of elements to process: i from 1 to LEN_2D-1
    num_elements = LEN_2D - 1
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
    )