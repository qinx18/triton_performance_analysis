import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on conditional increments
    # We need to handle the data-dependent indexing carefully
    
    # Process elements sequentially in blocks
    block_start = tl.program_id(0) * BLOCK_SIZE
    if block_start >= n_elements:
        return
    
    # Calculate how many elements this block should process
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Initialize j for this block
    # j starts at -1 globally, so for block starting at block_start, j starts at 2*block_start - 1
    j = 2 * block_start - 1
    
    # Process elements sequentially within the block
    for i_offset in range(BLOCK_SIZE):
        i = block_start + i_offset
        if i >= n_elements:
            break
            
        # j++ (first increment)
        j += 1
        
        # Load values for current i
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # if (c[i] > 0.0)
        if c_val > 0.0:
            j += 1
            # a[j] = c[i] + d[i] * e[i]
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # Loop goes to LEN_1D/2
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )