import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value in array a
    # Each program handles one block of elements
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of values
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find max in this block
    block_max = tl.max(vals)
    
    # Store block maximum
    tl.store(result_ptr + pid, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maxima
    block_maxima = torch.empty(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to find max in each block
    s314_kernel[(n_blocks,)](
        a, block_maxima, n_elements, BLOCK_SIZE
    )
    
    # Find global maximum from block maxima
    result = torch.max(block_maxima).item()
    
    return result