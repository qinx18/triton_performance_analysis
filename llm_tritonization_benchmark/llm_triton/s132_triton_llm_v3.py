import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < LEN_2D)
    
    # j = 0, k = 1 (since m = 0)
    j = 0
    k = 1
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load c[1] - broadcast to all elements
    c1_val = tl.load(c_ptr + 1)
    
    # Load aa[k][i-1] = aa[1][i-1]
    aa_k_offsets = k * LEN_2D + (offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1_val
    
    # Store to aa[j][i] = aa[0][i]
    aa_j_offsets = j * LEN_2D + offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa