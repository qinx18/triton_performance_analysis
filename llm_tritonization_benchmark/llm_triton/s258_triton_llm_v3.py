import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    aa_row0 = tl.load(aa_ptr + offsets, mask=mask)
    
    # Initialize s for this block
    s = 0.0
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            # Check condition for updating s
            if a[i] > 0.0:
                s = d[i] * d[i]
            
            # Compute outputs using current s
            b_val = s * c[i] + d[i]
            e_val = (s + 1.0) * aa_row0[i]
            
            # Store results
            tl.store(b_ptr + block_start + i, b_val)
            tl.store(e_ptr + block_start + i, e_val)

def s258_triton(a, b, c, d, e, aa):
    n_elements = a.shape[0]
    
    # Use block size of 1 to maintain sequential dependency of s
    BLOCK_SIZE = 1
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s258_kernel[grid](
        a, b, c, d, e, aa,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )