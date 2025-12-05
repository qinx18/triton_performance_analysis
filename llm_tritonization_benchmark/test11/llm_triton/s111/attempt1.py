import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Read from a[i-1] and b[i] 
    a_prev = tl.load(a_ptr + idx - 1)
    b_vals = tl.load(b_ptr + idx)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store to a[i]
    tl.store(a_ptr + idx, result)

def s111_triton(a, b):
    LEN_1D = a.shape[0]
    
    # Process elements from i=1 to LEN_1D-1 with step 2
    # This gives us indices: 1, 3, 5, 7, ...
    n_elements = (LEN_1D - 1 + 1) // 2  # Number of odd indices from 1
    
    # Due to loop-carried dependency, process one element at a time
    STRIP_SIZE = 1
    
    for strip_id in range(n_elements):
        # Calculate the actual array index (1, 3, 5, ...)
        strip_start = 1 + strip_id * 2
        
        s111_kernel[(1,)](
            a, b,
            strip_start,
            BLOCK_SIZE=STRIP_SIZE
        )