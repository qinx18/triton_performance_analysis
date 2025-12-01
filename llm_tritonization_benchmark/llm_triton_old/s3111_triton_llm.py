import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for conditional sum reduction - sum all positive elements.
    Uses block-level reduction with shared memory for efficiency.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking for out-of-bounds elements
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Conditional sum: only include positive values
    positive_mask = a_vals > 0.0
    conditional_vals = tl.where(positive_mask, a_vals, 0.0)
    
    # Block-level reduction sum
    block_sum = tl.sum(conditional_vals, axis=0)
    
    # Store partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    """
    Triton implementation of TSVC s3111 - conditional sum reduction.
    Performs block-wise reduction followed by final sum on CPU.
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output buffer for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel with 1D grid
    s3111_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU (small array)
    final_sum = torch.sum(partial_sums)
    
    return a