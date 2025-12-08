import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(
    aa_ptr, b_ptr, c_ptr,
    j, k, LEN_2D,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start (starting from i=1)
    block_start = pid * BLOCK_SIZE + 1
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid indices
    mask = i_offsets < LEN_2D
    
    # Load c[1]
    c_val = tl.load(c_ptr + 1)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load aa[k][i-1] = aa[k * LEN_2D + (i-1)]
    aa_k_indices = k * LEN_2D + (i_offsets - 1)
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_val
    
    # Store to aa[j][i] = aa[j * LEN_2D + i]
    aa_j_indices = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[1]
    
    # Number of elements to process (from i=1 to LEN_2D-1)
    n_elements = LEN_2D - 1
    
    # Block size
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(grid_size,)](
        aa, b, c,
        j, k, LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa