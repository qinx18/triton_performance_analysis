import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s261 with sequential dependency handling.
    Each thread block processes elements sequentially to maintain dependencies.
    """
    # Get program ID and calculate starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Process elements sequentially within each block
    for offset in range(BLOCK_SIZE):
        i = block_start + offset + 1  # Start from index 1
        
        # Boundary check
        if i >= n_elements:
            break
            
        # Load current elements
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)  # c[i-1] dependency
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # First computation: t = a[i] + b[i]; a[i] = t + c[i-1]
        t1 = a_val + b_val
        new_a = t1 + c_prev
        tl.store(a_ptr + i, new_a)
        
        # Second computation: t = c[i] * d[i]; c[i] = t
        t2 = c_val * d_val
        tl.store(c_ptr + i, t2)

def s261_triton(a, b, c, d):
    """
    Triton implementation of TSVC s261.
    Uses sequential processing to handle data dependencies.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Use small block size due to sequential dependencies
    # Each block processes elements sequentially to maintain correctness
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)  # -1 because we start from index 1
    
    # Launch kernel
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c