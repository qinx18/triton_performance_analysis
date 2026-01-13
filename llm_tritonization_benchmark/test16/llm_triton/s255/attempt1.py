import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < n_elements
    
    # Load x and y values (carry around variables)
    x = tl.load(b_ptr + (n_elements - 1))  # b[LEN_1D-1]
    y = tl.load(b_ptr + (n_elements - 2))  # b[LEN_1D-2]
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x + y) * 0.333
    
    # Store results
    tl.store(a_ptr + indices, result, mask=mask)

def s255_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s255_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )