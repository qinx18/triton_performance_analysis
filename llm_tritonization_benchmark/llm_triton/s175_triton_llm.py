import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(
    a_ptr,
    b_ptr,
    indices_ptr,
    n_elements,
    inc,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s175: a[i] = a[i + inc] + b[i] for strided indices
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load the actual indices to process
    indices = tl.load(indices_ptr + offsets, mask=mask)
    
    # Load a[indices + inc] and b[indices] with bounds checking
    # Create masks for memory safety
    a_read_mask = mask & (indices + inc < n_elements)  # Ensure we don't read beyond array bounds
    b_read_mask = mask & (indices >= 0)  # Ensure valid indices
    
    a_vals = tl.load(a_ptr + indices + inc, mask=a_read_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=b_read_mask, other=0.0)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_vals + b_vals
    
    # Store back to a[indices]
    store_mask = mask & (indices >= 0)
    tl.store(a_ptr + indices, result, mask=store_mask)

def s175_triton(a, b, inc):
    """
    Triton implementation of TSVC s175.
    
    Optimizations:
    - Pre-compute strided indices to avoid modulo operations in kernel
    - Use block-based processing for better memory coalescing
    - Explicit masking for memory safety and edge cases
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    
    # Generate indices for the strided loop (same as baseline)
    indices = torch.arange(0, len_1d - 1, inc, device=a.device, dtype=torch.long)
    n_indices = indices.numel()
    
    if n_indices == 0:
        return a
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_indices, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, indices,
        len_1d,  # Pass original array size for bounds checking
        inc,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a