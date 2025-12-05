import torch
import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, j, k, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the range of i values for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_vals = block_start + offsets + 1  # Start from i=1
    
    # Create mask for valid indices (i < LEN_2D)
    mask = i_vals < LEN_2D
    
    # Calculate pointers for aa[j][i] and aa[k][i-1]
    aa_j_ptrs = aa_ptr + j * LEN_2D + i_vals
    aa_k_ptrs = aa_ptr + k * LEN_2D + (i_vals - 1)
    
    # Load aa[k][i-1] values
    aa_k_vals = tl.load(aa_k_ptrs, mask=mask)
    
    # Load b[i] values
    b_ptrs = b_ptr + i_vals
    b_vals = tl.load(b_ptrs, mask=mask)
    
    # Load c[1] (scalar)
    c_1 = tl.load(c_ptr + 1)
    
    # Compute aa[j][i] = aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c_1
    
    # Store result
    tl.store(aa_j_ptrs, result, mask=mask)

def s132_triton(aa, b, c, j, k):
    LEN_2D = aa.shape[0]
    
    # Calculate grid size (number of i values to process is LEN_2D-1)
    n_elements = LEN_2D - 1
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(grid_size,)](
        aa, b, c, j, k, LEN_2D, BLOCK_SIZE
    )
    
    return aa