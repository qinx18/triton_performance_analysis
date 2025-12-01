import torch
import triton
import triton.language as tl

@triton.jit
def vag_kernel(
    a_ptr,
    b_ptr,
    ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for vectorized gather operation.
    Each thread block processes BLOCK_SIZE elements in parallel.
    """
    # Get program ID and calculate element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load indices from ip array with masking
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Gather operation: load b[ip[i]] values
    # Use indices to gather from b array
    gathered_values = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Store results to a array
    tl.store(a_ptr + offsets, gathered_values, mask=mask)

def vag_triton(a, b, ip):
    """
    Triton implementation of TSVC vag function.
    Optimized gather operation using GPU parallelization.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous() 
    ip = ip.contiguous()
    
    n_elements = a.numel()
    
    # Optimal block size for gather operations - balance memory coalescing and occupancy
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vag_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a