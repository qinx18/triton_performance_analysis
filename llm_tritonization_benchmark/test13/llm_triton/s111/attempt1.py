import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, start_idx, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = start_idx + offsets
    
    # Original loop: for (int i = 1; i < LEN_1D; i += 2)
    # So we need idx to represent the actual loop indices (1, 3, 5, ...)
    actual_idx = 1 + idx * 2
    
    mask = actual_idx < n_elements
    
    # Load a[i - 1] and b[i]
    a_prev = tl.load(a_ptr + actual_idx - 1, mask=mask)
    b_vals = tl.load(b_ptr + actual_idx, mask=mask)
    
    # Compute a[i] = a[i - 1] + b[i]
    result = a_prev + b_vals
    
    # Store to a[i]
    tl.store(a_ptr + actual_idx, result, mask=mask)

def s111_triton(a, b):
    LEN_1D = a.shape[0]
    
    # Original loop: for (int i = 1; i < LEN_1D; i += 2)
    # Number of iterations: (LEN_1D - 1 + 1) // 2 = LEN_1D // 2
    n_iterations = (LEN_1D - 1 + 1) // 2
    
    # Since we have loop-carried dependency, process one element at a time
    STRIP_SIZE = 1
    BLOCK_SIZE = 1
    
    # Process sequentially strip by strip
    for strip_start in range(0, n_iterations, STRIP_SIZE):
        s111_kernel[(1,)](
            a, b, 
            strip_start,
            LEN_1D,
            BLOCK_SIZE=BLOCK_SIZE
        )