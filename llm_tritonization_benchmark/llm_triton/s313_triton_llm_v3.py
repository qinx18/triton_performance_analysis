import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(
    a_ptr,
    b_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current block
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load elements from arrays a and b
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise product
    products = a_vals * b_vals
    
    # Sum the products within this block
    block_sum = tl.sum(products)
    
    # Store the block sum (each block writes one result)
    if pid == 0:  # Only first block initializes
        tl.store(result_ptr + pid, block_sum)
    else:
        # Atomic add for thread-safe accumulation
        tl.atomic_add(result_ptr, block_sum)

def s313_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Create result tensor
    dot = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s313_kernel[grid](
        a, b, dot,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dot.item()