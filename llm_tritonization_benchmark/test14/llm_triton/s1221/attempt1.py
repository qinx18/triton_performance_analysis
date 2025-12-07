import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4  # +4 because loop starts at i=4
    
    # Check bounds
    mask = idx < n_elements
    
    # Load from read positions
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + a_vals
    
    # Store to write positions
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    
    # Process elements from index 4 to n_elements-1
    start_idx = 4
    total_elements = n_elements - start_idx
    
    # Process in strips sequentially
    for strip_id in range(0, total_elements, STRIP_SIZE):
        strip_start = strip_id
        remaining = min(STRIP_SIZE, total_elements - strip_start)
        
        if remaining > 0:
            BLOCK_SIZE = triton.next_power_of_2(remaining)
            BLOCK_SIZE = min(BLOCK_SIZE, STRIP_SIZE)
            
            s1221_kernel[(1,)](
                b, a, 
                strip_start,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )