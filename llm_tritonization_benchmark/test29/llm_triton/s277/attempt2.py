import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must process sequentially due to b[i+1] = ... dependency
    # Use single thread to maintain sequential execution
    tid = tl.program_id(0)
    
    if tid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(n_elements - 1):
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] >= 0, if so skip to L20 (end of iteration)
        skip_to_l20 = a_val >= 0.0
        
        if not skip_to_l20:
            # Load b[i]
            b_val = tl.load(b_ptr + i)
            
            # Check if b[i] >= 0, if so skip to L30
            skip_to_l30 = b_val >= 0.0
            
            if not skip_to_l30:
                # Load c[i] and d[i] for a[i] update
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                
                # a[i] += c[i] * d[i]
                new_a_val = a_val + c_val * d_val
                tl.store(a_ptr + i, new_a_val)
        
        # L30: Always execute b[i+1] = c[i] + d[i] * e[i] unless we skipped to L20
        if not skip_to_l20:
            # Load values needed for b[i+1] calculation
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            result = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, result)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with only 1 block and 1 thread to maintain sequential execution
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements
    )
    
    return a, b