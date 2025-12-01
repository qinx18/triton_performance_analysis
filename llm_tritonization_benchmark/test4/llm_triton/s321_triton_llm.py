import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s321 - sequential dependency pattern.
    Each thread processes one element sequentially due to data dependency.
    """
    # This kernel processes elements sequentially due to dependency a[i] += a[i-1] * b[i]
    # We use a single thread block to maintain sequential access pattern
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block
    for i in range(1, n_elements):
        if tl.program_id(0) == 0:  # Only first block processes to maintain dependency
            if i < n_elements:
                # Load previous a value and current b value
                a_prev = tl.load(a_ptr + i - 1)
                b_curr = tl.load(b_ptr + i)
                a_curr = tl.load(a_ptr + i)
                
                # Update a[i] += a[i-1] * b[i]
                result = a_curr + a_prev * b_curr
                tl.store(a_ptr + i, result)

def s321_triton(a, b):
    """
    Triton implementation of TSVC s321 function.
    
    Sequential dependency pattern requires careful handling as each element
    depends on the previous computation result.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.shape[0]
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 128
    
    # Due to sequential dependency, we use a single grid
    grid = (1,)
    
    # Launch kernel
    s321_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a