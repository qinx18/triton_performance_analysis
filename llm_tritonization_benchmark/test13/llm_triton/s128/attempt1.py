import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel handles the coupled induction variables sequentially
    # Since j and k have dependencies, we process in blocks but maintain order
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Process elements in this block sequentially due to coupled induction variables
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        if i >= n_elements:
            break
            
        # Compute j and k values: j starts at -1, increments by 2 each iteration
        # j = -1, 1, 3, 5, ...
        # k = j + 1 = 0, 2, 4, 6, ...
        j = 2 * i - 1
        k = j + 1  # k = 2 * i
        
        # Load values
        d_val = tl.load(d_ptr + i)
        b_val = tl.load(b_ptr + k)
        c_val = tl.load(c_ptr + k)
        
        # First computation: a[i] = b[k] - d[i]
        a_val = b_val - d_val
        tl.store(a_ptr + i, a_val)
        
        # Second computation: b[k] = a[i] + c[k]
        b_new = a_val + c_val
        tl.store(b_ptr + k, b_new)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c, d