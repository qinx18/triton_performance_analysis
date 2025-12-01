import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get block ID and calculate offsets
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = 1 + block_id * BLOCK_SIZE + offsets
    
    # Mask for valid indices
    mask = i_offsets < LEN_2D
    
    # Load c[1] (scalar broadcast)
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # For aa[k][i-1] where k=1, i-1 gives us indices from 0 to LEN_2D-2
    # Load aa[1][i-1] values
    aa_k_offsets = LEN_2D + (i_offsets - 1)  # row 1, column i-1
    aa_k_vals = tl.load(aa_ptr + aa_k_offsets, mask=mask)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1] where j=0, k=1
    result = aa_k_vals + b_vals * c1
    
    # Store to aa[0][i] (j=0)
    aa_j_offsets = i_offsets  # row 0, column i
    tl.store(aa_ptr + aa_j_offsets, result, mask=mask)

def s132_triton(aa, b, c):
    LEN_2D = aa.shape[1]
    
    # Process indices from 1 to LEN_2D-1
    n_elements = LEN_2D - 1
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s132_kernel[grid](
        aa, b, c, LEN_2D, BLOCK_SIZE
    )