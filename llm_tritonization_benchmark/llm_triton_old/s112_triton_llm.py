import torch
import triton
import triton.language as tl

@triton.jit
def s112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s112 - Forward linear recurrence.
    Processes one iteration of the recurrence relation.
    """
    # Get program ID and compute block boundaries
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements - 1
    
    # Compute indices for the recurrence relation (reverse order)
    # Map block indices to reverse iteration order
    reverse_offsets = n_elements - 2 - offsets
    reverse_mask = reverse_offsets >= 0
    
    # Load data with masking
    a_vals = tl.load(a_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
    b_vals = tl.load(b_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
    
    # Compute recurrence: a[i+1] = a[i] + b[i]
    result = a_vals + b_vals
    
    # Store result at i+1 position
    store_offsets = reverse_offsets + 1
    store_mask = reverse_mask & (store_offsets < n_elements)
    tl.store(a_ptr + store_offsets, result, mask=store_mask)

def s112_triton(a, b, iterations):
    """
    Triton implementation of TSVC s112 - Forward linear recurrence.
    
    Uses sequential kernel launches to handle data dependencies.
    Each kernel processes one iteration of the outer loop.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.shape[0]
    
    # Handle edge case
    if n_elements <= 1:
        return a
    
    # Use smaller block size due to reverse memory access pattern
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements - 1, meta['BLOCK_SIZE']),)
    
    # Sequential execution of iterations to handle data dependencies
    for nl in range(3 * iterations):
        s112_kernel[grid](
            a, b, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a