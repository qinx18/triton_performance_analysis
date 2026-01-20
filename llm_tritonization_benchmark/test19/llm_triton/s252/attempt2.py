import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles one element sequentially
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one program processes all elements sequentially
        t = 0.0
        
        for i in range(n_elements):
            # Load single elements
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            
            # Compute
            s = b_val * c_val
            a_val = s + t
            t = s
            
            # Store result
            tl.store(a_ptr + i, a_val)

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    
    # Launch with single program since we need sequential processing
    grid = (1,)
    s252_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)