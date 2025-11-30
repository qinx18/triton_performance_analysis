import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate indices
    block_start = pid * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Local variables
    m = 0
    j = m  # j = 0
    k = m + 1  # k = 1
    
    # Load c[1]
    c_1 = tl.load(c_ptr + 1)
    
    # Load b[i] for current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load aa[k][i-1] = aa[1][i-1]
    aa_k_indices = k * LEN_2D + (offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    # aa[0][i] = aa[1][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store result to aa[j][i] = aa[0][i]
    aa_j_indices = j * LEN_2D + offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[0]
    
    # Block size for parallelization
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(num_blocks,)](
        aa, b, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )