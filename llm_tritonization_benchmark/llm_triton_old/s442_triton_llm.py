import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    indx = tl.load(indx_ptr + offsets, mask=mask)
    
    # Create masks for each switch case
    mask1 = indx == 1
    mask2 = indx == 2
    mask3 = indx == 3
    mask4 = indx == 4
    
    # Compute squared values
    b_sq = b * b
    c_sq = c * c
    d_sq = d * d
    e_sq = e * e
    
    # Apply conditional updates (vectorized switch statement)
    # Use tl.where to implement the branching logic
    result = a
    result = tl.where(mask1, result + b_sq, result)
    result = tl.where(mask2, result + c_sq, result)
    result = tl.where(mask3, result + d_sq, result)
    result = tl.where(mask4, result + e_sq, result)
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    """
    Triton implementation of TSVC s442 - optimized switch statement with conditional updates.
    
    Key optimizations:
    - Vectorized conditional operations using tl.where
    - Coalesced memory access patterns
    - Single kernel launch with appropriate block sizing
    - In-place updates to minimize memory traffic
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    indx = indx.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for good occupancy
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a