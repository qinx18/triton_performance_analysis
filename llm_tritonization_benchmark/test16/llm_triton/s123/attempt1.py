import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the conditional induction variable pattern
    # Each thread block handles one iteration sequentially due to dependencies
    
    block_id = tl.program_id(0)
    
    if block_id == 0:  # Only one block processes the entire computation
        j = 0  # Start j at 0 (will be incremented before first use)
        
        # Process elements sequentially
        for i in range(n_elements):
            # j++ (increment j)
            j = j + 1
            
            # Load scalar values for current i
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # a[j] = b[i] + d[i] * e[i]
            result1 = b_val + d_val * e_val
            tl.store(a_ptr + j, result1)
            
            # if (c[i] > 0.0)
            if c_val > 0.0:
                j = j + 1
                # a[j] = c[i] + d[i] * e[i]
                result2 = c_val + d_val * e_val
                tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2
    
    # Use single block since we have sequential dependencies
    BLOCK_SIZE = 1
    grid = (1,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a