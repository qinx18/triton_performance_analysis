import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # Each thread block processes one chunk of the iteration space
    block_id = tl.program_id(0)
    
    # Process iterations sequentially within each block
    block_start = block_id * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_half)
    
    for i in range(block_start, block_end):
        if i >= n_half:
            break
            
        # Load values for current i
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Calculate j position: j starts at -1, increments to 0, then conditionally to 1, etc.
        # For iteration i: j = 2*i if all previous c[k] > 0, otherwise less
        # We need to count how many times condition was met up to current i
        
        j = i  # Base j position (from j++ before first store)
        
        # Add count of previous positive c values
        for k in range(i):
            c_k = tl.load(c_ptr + k)
            if c_k > 0.0:
                j += 1
        
        # First store: a[j] = b[i] + d[i] * e[i]
        val1 = b_val + d_val * e_val
        tl.store(a_ptr + j, val1)
        
        # Conditional second store
        if c_val > 0.0:
            j += 1
            val2 = c_val + d_val * e_val
            tl.store(a_ptr + j, val2)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a