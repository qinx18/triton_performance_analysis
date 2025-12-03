import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction for the specific pattern in s31111
    # Each thread block handles the computation independently
    
    pid = tl.program_id(0)
    
    # We only need one thread block since this is a simple reduction
    if pid > 0:
        return
    
    # Load the required elements: a[0:4], a[4:8], a[8:12], a[12:16], a[16:20], a[20:24], a[24:28], a[28:32]
    sum_val = 0.0
    
    # Process each group of 4 elements (equivalent to test() function calls)
    for start_idx in range(0, 32, 4):
        offsets = tl.arange(0, 4) + start_idx
        mask = offsets < n_elements
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        group_sum = tl.sum(vals)
        sum_val += group_sum

def s31111_triton(a):
    # Get array size
    n_elements = a.numel()
    
    # Block size
    BLOCK_SIZE = 256
    
    # Grid size - only need one block for this simple computation
    grid = (1,)
    
    # Launch kernel
    s31111_kernel[grid](
        a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a