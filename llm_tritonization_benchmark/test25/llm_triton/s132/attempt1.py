import triton
import triton.language as tl
import torch

@triton.jit
def s132_kernel(
    aa_ptr,
    b_ptr, 
    c_ptr,
    len_2d,
    j, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Calculate the range of indices this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1  # Start from i=1
    
    # Create mask for valid indices (i < len_2d)
    mask = indices < len_2d
    
    # Load c[1]
    c1 = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Calculate aa[k][i-1] indices
    aa_k_indices = k * len_2d + (indices - 1)
    # Load aa[k][i-1] values
    aa_k_vals = tl.load(aa_ptr + aa_k_indices, mask=mask)
    
    # Compute: aa[k][i-1] + b[i] * c[1]
    result = aa_k_vals + b_vals * c1
    
    # Calculate aa[j][i] indices and store
    aa_j_indices = j * len_2d + indices
    tl.store(aa_ptr + aa_j_indices, result, mask=mask)

def s132_triton(aa, b, c, len_2d, j, k):
    # Get dimensions
    n_elements = len_2d - 1  # We process from i=1 to len_2d-1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s132_kernel[(n_blocks,)](
        aa,
        b,
        c,
        len_2d,
        j, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa