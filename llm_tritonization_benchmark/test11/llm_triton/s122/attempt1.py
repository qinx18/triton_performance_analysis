import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n_elements, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for idx in range(BLOCK_SIZE):
        global_idx = block_start + idx
        if global_idx >= n_elements:
            break
            
        # Compute the actual i value for this iteration
        i = n1 - 1 + global_idx * n3
        if i >= LEN_1D:
            break
            
        # Compute k value: k starts at 0, j=1, so k = j * (iteration+1) = iteration+1
        k = global_idx + 1
        
        # Load values
        b_idx = LEN_1D - k
        if b_idx >= 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + b_idx)
            tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations in the original loop
    n_elements = 0
    i = n1 - 1
    while i < LEN_1D:
        n_elements += 1
        i += n3
    
    if n_elements == 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n_elements, n1, n3, LEN_1D, BLOCK_SIZE
    )