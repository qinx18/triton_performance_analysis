import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks of the original loop
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load values from input arrays
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        
        # Calculate j indices: j starts at -1, then increments by 2 each iteration
        # For iteration i: j_first = 2*i, j_second = 2*i + 1
        j_first = 2 * idx
        j_second = 2 * idx + 1
        
        # Compute and store first result: a[j] = b[i] + c[i] * d[i]
        if j_first < 2 * n_elements:
            result1 = b_val + c_val * d_val
            tl.store(a_ptr + j_first, result1)
        
        # Compute and store second result: a[j] = b[i] + d[i] * e[i]
        if j_second < 2 * n_elements:
            result2 = b_val + d_val * e_val
            tl.store(a_ptr + j_second, result2)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )