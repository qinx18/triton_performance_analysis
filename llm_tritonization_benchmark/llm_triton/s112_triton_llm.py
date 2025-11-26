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
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 1)
    
    # Load from copy and b
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original a at offset+1
    write_offsets = offsets + 1
    write_mask = write_offsets < n_elements
    tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b, iterations):
    a = a.contiguous()
    b = b.contiguous()
    
    for nl in range(3 * iterations):
        # Create read-only copy for WAR dependency handling
        a_copy = a.clone()
        
        n_elements = a.numel()
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
        
        s112_kernel[grid](
            a,
            a_copy,
            b,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return a