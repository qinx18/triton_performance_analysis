import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential logic with data-dependent indexing
    # We need to process sequentially due to the conditional j increment
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Only use first thread block to maintain sequential semantics
    if pid != 0:
        return
    
    # Process all elements sequentially in first block
    j = -1
    
    # Process in chunks to avoid excessive memory usage
    for block_start in range(0, n_half, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_half)
        
        for i in range(block_start, block_end):
            j += 1
            
            # Load scalar values
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # First assignment: a[j] = b[i] + d[i] * e[i]
            result1 = b_val + d_val * e_val
            tl.store(a_ptr + j, result1)
            
            # Conditional assignment
            if c_val > 0.0:
                j += 1
                result2 = c_val + d_val * e_val
                tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to maintain sequential semantics
    grid = (1,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )