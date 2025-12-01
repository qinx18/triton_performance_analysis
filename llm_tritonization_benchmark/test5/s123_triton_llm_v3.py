import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential processing within each block
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        if i >= n_elements:
            break
            
        # Load scalar values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Calculate output index j (2*i for first write, 2*i+1 for conditional write)
        j = 2 * i
        
        # First write: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Conditional write: if (c[i] > 0) then a[j+1] = c[i] + d[i] * e[i]
        if c_val > 0.0:
            j_next = j + 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j_next, result2)

def s123_triton(a, b, c, d, e):
    n_elements = len(b) // 2  # LEN_1D/2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )