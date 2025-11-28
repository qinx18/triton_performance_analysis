import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    # Compute product reduction within this block
    prod = tl.reduce(a_vals, 0, tl.reduce_op.MUL)
    
    # Store the partial product (only first thread in block writes)
    if tl.program_id(0) == 0 and offsets[0] == block_start:
        tl.store(output_ptr + block_id, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Create output tensor for partial products
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_products = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial products
    s312_kernel[grid](
        a,
        partial_products,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction on CPU/GPU
    prod = torch.prod(partial_products)
    
    return prod