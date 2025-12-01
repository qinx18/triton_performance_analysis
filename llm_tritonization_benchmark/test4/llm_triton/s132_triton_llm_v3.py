import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate the range of indices this block will handle
    block_start = pid * BLOCK_SIZE + 1  # Start from 1 as per original loop
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask to ensure we don't go beyond LEN_2D
    mask = i_offsets < LEN_2D
    
    # Load c[1] (scalar broadcast)
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load aa[1][i-1] values (k=1, i-1 indices)
    aa_read_offsets = LEN_2D + (i_offsets - 1)  # aa[1][i-1] in flattened format
    aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=mask)
    
    # Compute the result
    result = aa_vals + b_vals * c1
    
    # Store to aa[0][i] (j=0, i indices)
    aa_write_offsets = i_offsets  # aa[0][i] in flattened format
    tl.store(aa_ptr + aa_write_offsets, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[1]
    
    # Calculate number of elements to process (from 1 to LEN_2D-1)
    n_elements = LEN_2D - 1
    
    # Block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(grid_size,)](
        aa, b, c,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa