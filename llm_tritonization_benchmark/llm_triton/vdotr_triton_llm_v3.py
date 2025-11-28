import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block computes a partial dot product
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial dot product
    partial_dot = tl.sum(a_vals * b_vals)
    
    # Store partial result
    block_id = tl.program_id(0)
    tl.store(output_ptr + block_id, partial_dot)

def vdotr_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    vdotr_kernel[(grid_size,)](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums).item()
    return dot