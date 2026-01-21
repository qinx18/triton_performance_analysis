import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from i=1
    i_indices = offsets + 1
    mask = i_indices < LEN_2D
    
    # Load c[1] (scalar broadcast)
    c1_ptr = c_ptr + 1
    c1_val = tl.load(c1_ptr)
    
    # Load b[i] values
    b_ptrs = b_ptr + i_indices
    b_vals = tl.load(b_ptrs, mask=mask)
    
    # Load aa[k][i-1] values (i-1 because we start from i=1)
    aa_k_ptrs = aa_ptr + k * LEN_2D + (i_indices - 1)
    aa_k_vals = tl.load(aa_k_ptrs, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i]
    aa_j_ptrs = aa_ptr + j * LEN_2D + i_indices
    tl.store(aa_j_ptrs, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # We process indices from 1 to LEN_2D-1, so total elements = LEN_2D-1
    num_elements = LEN_2D - 1
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
    )