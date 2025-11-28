import triton
import triton.language as tl
import torch

@triton.jit
def s176_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for j dimension
    j = tl.program_id(0)
    
    # Load c[j] once for this program
    c_val = tl.load(c_ptr + j)
    
    # Process i dimension in blocks
    i_block_start = tl.program_id(1) * BLOCK_SIZE
    i_offsets = i_block_start + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < m
    
    # Load a[i] values from the read-only copy
    a_vals = tl.load(a_copy_ptr + i_offsets, mask=i_mask)
    
    # Calculate b indices: i + m - j - 1
    b_offsets = i_offsets + m - j - 1
    b_vals = tl.load(b_ptr + b_offsets, mask=i_mask)
    
    # Compute: a[i] += b[i+m-j-1] * c[j]
    result = a_vals + b_vals * c_val
    
    # Store result back to original a array
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    m = LEN_1D // 2
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Grid configuration
    BLOCK_SIZE = 256
    grid = (
        m,  # j dimension (0 to m-1)
        triton.cdiv(m, BLOCK_SIZE),  # i dimension blocks
    )
    
    # Launch kernel for each j value
    for j in range(m):
        s176_kernel[(1, triton.cdiv(m, BLOCK_SIZE))](
            a,
            a_copy,
            b,
            c,
            m=m,
            BLOCK_SIZE=BLOCK_SIZE,
        )