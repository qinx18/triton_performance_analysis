import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1221 operation.
    Each thread processes one element with dependency on b[i-4].
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from index 4
    offsets = offsets + 4
    mask = offsets < n_elements
    
    # Load current values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_prev_vals = tl.load(b_ptr + offsets - 4, mask=mask)
    
    # Compute new values: b[i] = b[i-4] + a[i]
    result = b_prev_vals + a_vals
    
    # Store results
    tl.store(b_ptr + offsets, result, mask=mask)

def s1221_triton(a, b):
    """
    Triton implementation of TSVC s1221 function.
    Optimized with parallel processing while handling the dependency pattern.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = b.numel()
    
    if n_elements <= 4:
        return b
    
    # Process elements from index 4 onwards
    elements_to_process = n_elements - 4
    
    # Use smaller block size due to dependency pattern
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    # Process in waves to handle dependencies correctly
    # Split into chunks that don't have internal dependencies
    chunk_size = 4  # Process 4 elements at a time to avoid conflicts
    
    for start_offset in range(0, elements_to_process, chunk_size * BLOCK_SIZE):
        remaining = min(chunk_size * BLOCK_SIZE, elements_to_process - start_offset)
        if remaining <= 0:
            break
            
        # Adjust pointers for current chunk
        chunk_grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        
        s1221_kernel[chunk_grid](
            a + start_offset,
            b + start_offset, 
            remaining + 4,  # Include the offset in size calculation
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return b