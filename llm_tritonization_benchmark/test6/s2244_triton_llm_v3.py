import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block to maintain dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load values for current index
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            e_val = tl.load(e_ptr + idx)
            
            # First statement: a[i+1] = b[i] + e[i]
            if idx + 1 < n_elements + 1:  # Check bounds for i+1
                tl.store(a_ptr + idx + 1, b_val + e_val)
            
            # Second statement: a[i] = b[i] + c[i]
            tl.store(a_ptr + idx, b_val + c_val)

def s2244_triton(a, b, c, e):
    n_elements = len(b) - 1  # Loop goes to LEN_1D-1
    
    # Use small block size to minimize dependency issues
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a