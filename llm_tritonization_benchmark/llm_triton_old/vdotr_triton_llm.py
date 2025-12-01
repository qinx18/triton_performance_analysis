import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load elements with masking
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise multiplication
    products = a * b
    
    # Sum within block using reduction
    block_sum = tl.sum(products, axis=0)
    
    # Store partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def vdotr_triton(a, b):
    """
    Triton implementation of TSVC vdotr function.
    Optimized dot product computation using block-wise reduction.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Use power of 2 block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for partial sums
    partial_sums = torch.empty(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial dot products
    vdotr_kernel[(grid_size,)](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction on CPU/GPU to get scalar result
    dot = torch.sum(partial_sums)
    
    return dot