import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(
    a_ptr, b_ptr, ip_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s4115 - indirect addressing with dot product.
    Each block processes BLOCK_SIZE elements and accumulates partial sums.
    """
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load a values with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load indirect indices with masking
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute element-wise products and sum within block
    products = a_vals * b_vals
    block_sum = tl.sum(products)
    
    # Store partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    """
    Triton implementation of TSVC s4115 - indirect addressing with dot product.
    Uses block-based processing with reduction for GPU optimization.
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    n_elements = a.shape[0]
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros((grid_size,), device=a.device, dtype=a.dtype)
    
    # Launch kernel to compute partial sums
    s4115_kernel[(grid_size,)](
        a, b, ip, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction of partial sums (done on GPU)
    sum_val = torch.sum(partial_sums)
    
    return a, b, ip