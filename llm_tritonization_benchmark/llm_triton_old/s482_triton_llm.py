import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(
    a_ptr, b_ptr, c_ptr, break_idx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s482: a[i] += b[i] * c[i] with early break on c[i] > b[i]
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load break index (computed on CPU)
    break_idx = tl.load(break_idx_ptr)
    
    # Create mask for valid elements within break point
    mask = (offsets < n_elements) & (offsets < break_idx)
    
    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute update: a[i] += b[i] * c[i]
    result = a + b * c
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s482_triton(a, b, c):
    """
    Triton implementation of TSVC s482 function.
    Optimized GPU kernel with early break condition handling.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Find break index on GPU, then move to CPU for kernel parameter
    condition = c > b
    if torch.any(condition):
        break_idx = torch.argmax(condition.int()).item()
    else:
        break_idx = n_elements
    
    # Create break_idx tensor for kernel
    break_idx_tensor = torch.tensor([break_idx], device=a.device, dtype=torch.int32)
    
    # Early exit if break_idx is 0
    if break_idx == 0:
        return a
    
    # Launch Triton kernel with optimal block size for memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(min(break_idx, n_elements), BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c, break_idx_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a