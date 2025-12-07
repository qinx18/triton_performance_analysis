import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    tid = tl.program_id(0)
    
    if tid != 0:
        return
    
    # Process sequentially from index 2 to n_elements-1
    for i in range(2, n_elements):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result back to a[i]
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Use single thread since computation must be sequential
    BLOCK_SIZE = 1
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a