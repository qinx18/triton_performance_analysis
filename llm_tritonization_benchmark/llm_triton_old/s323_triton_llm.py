import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s323 operation.
    Each thread block processes BLOCK_SIZE elements sequentially to maintain dependencies.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Process elements sequentially within each block to handle dependencies
    for offset in range(BLOCK_SIZE):
        i = block_start + offset + 1  # Start from index 1
        
        # Boundary check
        if i >= n_elements:
            break
            
        # Load values
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i] (using just computed a_val)
        b_val = a_val + c_val * e_val
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s323 function.
    Uses sequential processing within blocks to handle data dependencies.
    """
    n_elements = a.shape[0]
    
    # Use smaller block size due to sequential dependencies
    BLOCK_SIZE = 64
    grid = lambda meta: (triton.cdiv(n_elements - 1, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b