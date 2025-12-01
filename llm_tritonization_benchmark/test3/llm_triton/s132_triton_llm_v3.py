import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the range of i values this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (i starts from 1, not 0)
    mask = (offsets >= 1) & (offsets < LEN_2D)
    
    # Local variables from C code
    m = 0
    j = m      # j = 0
    k = m + 1  # k = 1
    
    # Load c[1] (scalar value)
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Load aa[k][i-1] = aa[1][i-1] values
    # Need to be careful with i-1 indexing
    aa_read_offsets = k * LEN_2D + (offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_read_offsets, mask=mask, other=0.0)
    
    # Compute: aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1
    
    # Store to aa[j][i] = aa[0][i]
    aa_write_offsets = j * LEN_2D + offsets
    tl.store(aa_ptr + aa_write_offsets, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[1]
    
    # Block size for parallelization
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(num_blocks,)](
        aa, b, c,
        LEN_2D,
        BLOCK_SIZE,
    )
    
    return aa