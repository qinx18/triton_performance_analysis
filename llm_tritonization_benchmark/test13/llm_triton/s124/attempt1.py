import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially to maintain the induction variable j
    for i in range(n_elements):
        # Each thread block processes one element at a time to maintain order
        if tl.program_id(0) == 0:
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # Compute the result based on condition
            result = tl.where(b_val > 0.0, b_val + d_val * e_val, c_val + d_val * e_val)
            
            # Store at position j (which equals i since j starts at -1 and increments each iteration)
            tl.store(a_ptr + i, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with only one block to maintain sequential execution
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a