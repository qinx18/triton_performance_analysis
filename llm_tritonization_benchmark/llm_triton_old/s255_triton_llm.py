import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s255 computation.
    Each thread block processes a chunk of the array sequentially.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load x and y initial values (shared across all blocks)
    x = tl.load(b_ptr + n_elements - 1)
    y = tl.load(b_ptr + n_elements - 2)
    
    # Process elements in this block sequentially to maintain dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load current b[i]
            b_val = tl.load(b_ptr + idx)
            
            # Compute result: (b[i] + x + y) * 0.333
            result = (b_val + x + y) * 0.333
            
            # Store result
            tl.store(a_ptr + idx, result)
            
            # Update x and y for next iteration
            y = x
            x = b_val

def s255_triton(a, b):
    """
    Triton implementation of TSVC s255.
    
    Key optimizations:
    - Uses sequential processing within blocks to handle data dependencies
    - Minimizes memory access overhead with efficient loading patterns
    - Uses masking for edge case handling
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = b.size(0)
    
    # Use smaller block size due to sequential dependencies
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel - note: must use num_warps=1 to ensure sequential execution
    s255_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    
    return a