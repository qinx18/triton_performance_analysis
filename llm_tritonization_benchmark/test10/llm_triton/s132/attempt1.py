import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j: tl.constexpr, k: tl.constexpr, 
                LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1  # i starts from 1
    
    mask = i_offsets < LEN_2D
    
    # Load aa[k][i-1] values
    aa_k_offsets = k * LEN_2D + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load c[1] - scalar broadcast
    c_1_offsets = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    c_1_val = tl.load(c_ptr + c_1_offsets, mask=tl.full([BLOCK_SIZE], True, dtype=tl.int1))
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1_val
    
    # Store to aa[j][i]
    aa_j_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[1]
    n_elements = LEN_2D - 1  # Loop from 1 to LEN_2D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, j, k,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa