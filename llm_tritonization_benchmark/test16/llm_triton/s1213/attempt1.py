import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel must be sequential due to loop-carried dependency
    # b[i] = a[i+1] * d[i] depends on a[i+1] which may be written in same iteration
    # Process one element per thread sequentially
    
    tid = tl.program_id(0)
    
    if tid == 0:  # Only use one thread for sequential processing
        for i in range(1, n_elements - 1):
            # a[i] = b[i-1] + c[i]
            b_val = tl.load(b_ptr + (i - 1))
            c_val = tl.load(c_ptr + i)
            a_val = b_val + c_val
            tl.store(a_ptr + i, a_val)
            
            # b[i] = a[i+1] * d[i]
            # Use original a values for a[i+1]
            a_next_val = tl.load(a_copy_ptr + (i + 1))
            d_val = tl.load(d_ptr + i)
            b_val = a_next_val * d_val
            tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Only one block since we need sequential processing
    
    s1213_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE,
    )