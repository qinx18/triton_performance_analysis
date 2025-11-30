import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate the range of indices this block will handle
    start_idx = pid * BLOCK_SIZE + 1  # Start from 1
    block_indices = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go beyond LEN_2D
    mask = block_indices < LEN_2D
    
    # Load c[1] (scalar, broadcasted)
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + block_indices, mask=mask)
    
    # Calculate aa[k][i-1] addresses (k=1, so row 1, columns i-1)
    # aa[1][i-1] where i-1 ranges from 0 to LEN_2D-2
    aa_k_addrs = LEN_2D + (block_indices - 1)  # row 1 * LEN_2D + (i-1)
    aa_k_vals = tl.load(aa_ptr + aa_k_addrs, mask=mask)
    
    # Compute the result
    result = aa_k_vals + b_vals * c1
    
    # Store to aa[j][i] where j=0, so row 0
    aa_j_addrs = block_indices  # row 0 * LEN_2D + i
    tl.store(aa_ptr + aa_j_addrs, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)  # -1 because we start from index 1
    
    s132_kernel[(num_blocks,)](
        aa, b, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa