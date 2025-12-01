import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s244 computation with sequential dependencies.
    Each thread block processes a contiguous chunk sequentially.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Process elements sequentially within each block to maintain dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        # Check bounds - process up to n_elements-1 as per original loop
        if idx >= n_elements - 1:
            break
            
        # Load values
        a_i = tl.load(a_ptr + idx)
        a_i_plus_1 = tl.load(a_ptr + idx + 1)
        b_i = tl.load(b_ptr + idx)
        c_i = tl.load(c_ptr + idx)
        d_i = tl.load(d_ptr + idx)
        
        # Compute updates following original order
        new_a_i = b_i + c_i * d_i
        new_b_i = c_i + b_i
        new_a_i_plus_1 = new_b_i + a_i_plus_1 * d_i
        
        # Store results
        tl.store(a_ptr + idx, new_a_i)
        tl.store(b_ptr + idx, new_b_i)
        tl.store(a_ptr + idx + 1, new_a_i_plus_1)

def s244_triton(a, b, c, d):
    """
    Triton implementation of TSVC s244 function.
    Uses sequential processing within blocks to handle data dependencies.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    LEN_1D = a.shape[0]
    
    if LEN_1D <= 1:
        return a, b
    
    # Use smaller block size due to sequential dependencies
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(LEN_1D - 1, BLOCK_SIZE)
    
    s244_kernel[(grid_size,)](
        a, b, c, d,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b