import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    if block_start >= n_elements:
        return
    
    # Process elements sequentially within each block to maintain dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load current values
        e_val = tl.load(e_ptr + idx)
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        
        # Get previous s value (stored in a[idx-1] if idx > block_start, else 0)
        if idx == 0:
            s = 0.0
        else:
            s = tl.load(b_ptr + (idx - 1)) + tl.load(c_ptr + (idx - 1))
            if idx > block_start:
                # For elements within block, use the updated value
                s = tl.load(b_ptr + (idx - 1)) + tl.load(c_ptr + (idx - 1))
        
        # Compute a[i] = s * e[i]
        a_val = s * e_val
        tl.store(a_ptr + idx, a_val)
        
        # Compute s = b[i] + c[i] for next iteration
        s = b_val + c_val
        
        # Compute b[i] = a[i] + d[i]
        b_new = a_val + d_val
        tl.store(b_ptr + idx, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Use small block size to maintain sequential dependencies
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with sequential processing
    for block_id in range(grid[0]):
        s2251_kernel[(1,)](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )