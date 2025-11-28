import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the range of i values this block will handle
    block_start = pid * BLOCK_SIZE + 1  # Start from 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Local variables
    m = 0
    j = m  # j = 0
    k = m + 1  # k = 1
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Load aa[k][i-1] = aa[1][i-1]
    # For i-1 indexing, we need to ensure i-1 >= 0, but since we start from i=1, i-1>=0
    aa_k_indices = k * LEN_2D + (offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask, other=0.0)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store to aa[j][i] = aa[0][i]
    aa_j_indices = j * LEN_2D + offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[0]
    
    # Block size for parallelization
    BLOCK_SIZE = 256
    
    # Number of blocks needed (excluding i=0)
    num_elements = LEN_2D - 1
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    s132_kernel[(grid_size,)](
        aa, b, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa