import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(
    a_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for vector sum reduction.
    Uses block-level reduction for efficient GPU utilization.
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data with masking (zeros for out-of-bounds)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Perform block-level sum reduction
    block_sum = tl.sum(a_vals)
    
    # Store partial sum for this block
    tl.store(sum_ptr + pid, block_sum)

def vsumr_triton(a):
    """
    Triton implementation of TSVC vsumr function.
    
    Optimizations:
    - Block-level parallel reduction
    - Coalesced memory access
    - Efficient GPU memory utilization
    """
    a = a.contiguous()
    
    # Kernel configuration
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary buffer for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel for parallel reduction
    vsumr_kernel[(grid_size,)](
        a,
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction of partial sums on GPU
    sum_val = torch.sum(partial_sums)
    
    return a