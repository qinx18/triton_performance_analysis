import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(e_ptr, BLOCK_SIZE: tl.constexpr, n_elements):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load initial value e[0]
    e0 = tl.load(e_ptr)
    
    # Process blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Skip if this is the first element (i=0)
        valid_mask = mask & (current_offsets > 0)
        
        # Compute 2^i for each position
        exponents = tl.exp2(current_offsets.to(tl.float32))
        
        # Compute e[i] = e0^(2^i)
        result = tl.exp(tl.log(e0) * exponents)
        
        # Store results
        tl.store(e_ptr + current_offsets, result, mask=valid_mask)

def s222_triton(a, b, c, e):
    N = a.shape[0]
    
    # Skip operations on array 'a' since they cancel out completely
    # a[i] += b[i] * c[i] followed by a[i] -= b[i] * c[i] = no change
    
    # Only process the power recurrence for array 'e'
    # Transform e[i] = e[i-1]^2 into parallel computation e[i] = e[0]^(2^i)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s222_kernel[grid](
        e,
        BLOCK_SIZE=BLOCK_SIZE,
        n_elements=N
    )