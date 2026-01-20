import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel implements sequential computation where s accumulates
    # Each block handles BLOCK_SIZE consecutive elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize s for this block
    s = 0.0
    
    # Each block needs to know the accumulated s from previous blocks
    # s = 2.0 * block_start at the start of this block
    s = 2.0 * block_start
    
    # Process elements in this block
    mask = (block_start + offsets) < n
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # For each element in the block, compute s and a[i]
    # Since s increments by 2.0 for each i: s = 2.0 * (block_start + i + 1)
    element_indices = block_start + offsets + 1
    s_vals = 2.0 * element_indices
    
    # Compute a[i] = s * b[i]
    a_vals = s_vals * b_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)

def s453_triton(a, b):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )