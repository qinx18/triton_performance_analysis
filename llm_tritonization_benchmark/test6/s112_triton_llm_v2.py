import torch
import triton
import triton.language as tl

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements - 1
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i+1] = a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to original array at i+1 positions
    write_offsets = offsets + 1
    write_mask = write_offsets < n_elements
    tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b, iterations):
    a = a.contiguous()
    b = b.contiguous()
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    for nl in range(3 * iterations):
        # Create read-only copy for this iteration
        a_copy = a.clone()
        
        # Launch kernel
        grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
        s112_kernel[grid](
            a,
            a_copy,
            b,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return a