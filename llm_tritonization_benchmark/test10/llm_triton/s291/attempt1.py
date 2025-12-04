import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load b values for current block
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # For each element in the block, we need to compute:
    # a[i] = (b[i] + b[im1]) * 0.5 where im1 is the previous index
    
    # Handle the sequential dependency by processing element by element
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i
        if global_i >= n_elements:
            break
            
        # Calculate im1 (previous index)
        if global_i == 0:
            im1 = n_elements - 1
        else:
            im1 = global_i - 1
            
        # Load b[i] and b[im1]
        b_i = tl.load(b_ptr + global_i)
        b_im1 = tl.load(b_ptr + im1)
        
        # Compute result
        result = (b_i + b_im1) * 0.5
        
        # Store result
        tl.store(a_ptr + global_i, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s291_kernel[grid](
        a, b, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )