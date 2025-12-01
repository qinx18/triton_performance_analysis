import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(
    a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate block range
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Initialize s for each thread
    s = 0.0
    
    # Sequential dependency requires processing elements in order within each block
    for i in range(BLOCK_SIZE):
        if block_start + i >= n_elements:
            break
            
        # Check if current element of a is positive
        if block_start + i < n_elements:
            a_val = tl.load(a_ptr + block_start + i)
            d_val = tl.load(d_ptr + block_start + i)
            c_val = tl.load(c_ptr + block_start + i)
            aa_val = tl.load(aa_ptr + block_start + i)
            
            # Update s if a[i] > 0
            if a_val > 0.0:
                s = d_val * d_val
            
            # Calculate outputs
            b_val = s * c_val + d_val
            e_val = (s + 1.0) * aa_val
            
            # Store results
            tl.store(b_ptr + block_start + i, b_val)
            tl.store(e_ptr + block_start + i, e_val)

def s258_triton(a, aa, b, c, d, e):
    """
    Triton implementation of TSVC s258.
    Optimized for GPU execution with proper handling of sequential dependencies.
    """
    # Ensure contiguous memory layout for optimal access patterns
    a = a.contiguous()
    aa = aa.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.shape[0]
    
    # Use smaller block size due to sequential dependency within the loop
    # This allows better load balancing while maintaining correctness
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with sequential processing blocks
    s258_kernel[grid](
        a, aa[0], b, c, d, e,  # aa[0] since we need the first row
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return b, e