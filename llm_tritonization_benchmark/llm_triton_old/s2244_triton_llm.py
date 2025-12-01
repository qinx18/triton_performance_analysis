import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(
    a_ptr, b_ptr, c_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s2244 operation.
    Key optimization: Process elements in parallel while handling dependencies.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements (n_elements - 1 since loop goes to len_1d - 1)
    mask = offsets < (n_elements - 1)
    
    # Load input data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute a[i+1] = b[i] + e[i] (store at offset+1)
    a_next_vals = b_vals + e_vals
    tl.store(a_ptr + offsets + 1, a_next_vals, mask=mask)
    
    # Compute a[i] = b[i] + c[i] (store at offset)
    a_curr_vals = b_vals + c_vals
    tl.store(a_ptr + offsets, a_curr_vals, mask=mask)

def s2244_triton(a, b, c, e):
    """
    Triton implementation of TSVC s2244
    
    Optimizations:
    - Parallel processing of independent computations
    - Efficient memory coalescing with contiguous access patterns
    - Single kernel launch to minimize overhead
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    e = e.contiguous()
    
    len_1d = a.size(0)
    
    # Process n_elements - 1 elements (matching the original loop bound)
    n_elements = len_1d
    
    # Use block size that balances occupancy and memory efficiency
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a