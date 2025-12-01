import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This is a reduction kernel - product of all elements in array a
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    # Compute product within this block
    block_prod = tl.reduce(a_vals, axis=0, combine_fn=lambda x, y: x * y)
    
    # Store partial product for this block
    tl.store(output_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.numel()
    
    # Calculate number of blocks needed
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial products
    partial_products = torch.ones(n_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel to compute partial products
    s312_kernel[(n_blocks,)](
        a,
        partial_products,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction on CPU/GPU to get single product
    prod = torch.prod(partial_products)
    
    return prod.item()