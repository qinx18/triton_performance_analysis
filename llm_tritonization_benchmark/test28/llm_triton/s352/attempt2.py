import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    block_sum = tl.sum(products)
    
    tl.store(partial_sums_ptr + pid, block_sum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create temporary array for partial sums
    partial_sums = torch.zeros(grid[0], dtype=torch.float32, device=a.device)
    
    s352_kernel[grid](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results on CPU
    return partial_sums.sum().item()