import torch
import triton
import triton.language as tl

@triton.jit
def s279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    a_out_ptr, b_out_ptr, c_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition for first branch (a[i] > 0)
    cond1 = a > 0.0
    
    # Path when a[i] <= 0: compute new b value
    b_new = -b + d * d
    cond2 = b_new <= a
    
    # Compute different c paths based on conditions
    # When a[i] > 0: c[i] = -c[i] + e[i] * e[i]
    c_path1 = -c + e * e
    
    # When a[i] <= 0 and b[i] > a[i]: c[i] += d[i] * e[i]
    c_path2 = c + d * e
    
    # When a[i] <= 0 and b[i] <= a[i]: c[i] remains unchanged
    c_path3 = c
    
    # Select the appropriate c value using nested where logic
    c_updated = tl.where(cond1, c_path1, tl.where(cond2, c_path3, c_path2))
    
    # Update b only when a[i] <= 0
    b_updated = tl.where(cond1, b, b_new)
    
    # Final update: a[i] = b[i] + c[i] * d[i]
    a_updated = b_updated + c_updated * d
    
    # Store results
    tl.store(a_out_ptr + offsets, a_updated, mask=mask)
    tl.store(b_out_ptr + offsets, b_updated, mask=mask)
    tl.store(c_out_ptr + offsets, c_updated, mask=mask)

def s279_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s279 - control flow with goto statements
    Optimized for GPU with coalesced memory access and vectorized operations
    """
    # Ensure contiguous memory layout for optimal access patterns
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Create output tensors
    a_out = torch.empty_like(a)
    b_out = torch.empty_like(b)
    c_out = torch.empty_like(c)
    
    # Use block size optimized for memory coalescing and occupancy
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimal grid configuration
    s279_kernel[grid](
        a, b, c, d, e,
        a_out, b_out, c_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_out, b_out, c_out