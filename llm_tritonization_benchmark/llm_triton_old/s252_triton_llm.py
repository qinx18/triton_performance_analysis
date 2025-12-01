import torch
import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s252 - scalar and array expansion with sequential dependency
    Uses single thread to maintain sequential dependency where a[i] depends on previous computation
    """
    pid = tl.program_id(axis=0)
    
    # Only use first thread to maintain sequential dependency
    if pid != 0:
        return
    
    # Sequential computation to maintain dependency t = s from previous iteration
    t = 0.0
    
    # Process elements in blocks while maintaining sequential dependency
    block_start = 0
    while block_start < n_elements:
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        actual_block_size = block_end - block_start
        
        # Load block of b and c values
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_end
        
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(actual_block_size):
            s = b_vals[i] * c_vals[i]
            a_val = s + t
            # Store single element
            tl.store(a_ptr + block_start + i, a_val)
            t = s
        
        block_start = block_end

def s252_triton(a, b, c):
    """
    Triton implementation of TSVC s252 - scalar and array expansion
    Maintains sequential dependency where each a[i] depends on previous computation
    """
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 256
    
    # Launch single program to maintain sequential dependency
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a