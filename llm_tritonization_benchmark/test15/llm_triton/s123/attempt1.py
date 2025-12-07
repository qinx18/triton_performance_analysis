import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize j for this block
    j = block_start * 2 - 1
    
    # Process elements in this block
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        if i >= n_elements:
            break
            
        j += 1
        
        # Load values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Conditional assignment
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Reset j and launch kernel
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )