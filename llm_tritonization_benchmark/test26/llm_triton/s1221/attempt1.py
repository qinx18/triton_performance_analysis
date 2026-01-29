import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, n_elements, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = (idx < n_elements) & (idx >= 4)
    
    # Load from read position (idx - 4) and write position (idx)
    read_idx = idx - 4
    b_vals = tl.load(b_ptr + read_idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = b_vals + a_vals
    
    # Store to write position
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    # Process in strips sequentially
    for strip_start in range(4, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_block_size = strip_end - strip_start
        
        if current_block_size > 0:
            # Launch kernel with single block for this strip
            s1221_kernel[(1,)](
                b, a, n_elements, strip_start, BLOCK_SIZE=BLOCK_SIZE
            )