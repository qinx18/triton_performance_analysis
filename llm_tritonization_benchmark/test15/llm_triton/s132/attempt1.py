import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    j, k, LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1  # Start from i=1
    
    # Create mask for valid indices
    mask = i_offsets < LEN_2D
    
    # Load b[i] values
    b_offsets = i_offsets
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    # Load c[1] (scalar)
    c_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] values
    aa_k_offsets = k * LEN_2D + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask, other=0.0)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_val
    
    # Store result to aa[j][i]
    aa_j_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[1]
    n_elements = LEN_2D - 1  # Process from i=1 to LEN_2D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c,
        j, k, LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )