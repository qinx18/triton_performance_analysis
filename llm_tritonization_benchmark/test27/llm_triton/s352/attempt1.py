import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Initialize dot product accumulator for this block
    dot_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute dot product for each element
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication
    products = a_vals * b_vals
    
    # Sum reduction within block
    dot_sum = tl.sum(products)
    
    # Store result (only first thread writes the sum)
    if tl.program_id(0) == 0:
        tl.atomic_add(result_ptr, dot_sum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Initialize result tensor
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s352_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()