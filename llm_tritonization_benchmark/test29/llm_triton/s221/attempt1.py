import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        # Check bounds
        if idx >= n_elements:
            break
        if idx < 1:  # Loop starts at i=1
            continue
            
        # Load values
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        a_val = tl.load(a_ptr + idx)
        
        # First statement: a[i] += c[i] * d[i]
        new_a_val = a_val + c_val * d_val
        tl.store(a_ptr + idx, new_a_val)
        
        # Second statement: b[i] = b[i-1] + a[i] + d[i]
        b_prev = tl.load(b_ptr + idx - 1)
        new_b_val = b_prev + new_a_val + d_val
        tl.store(b_ptr + idx, new_b_val)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s221_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b