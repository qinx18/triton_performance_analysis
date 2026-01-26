import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n, BLOCK_SIZE):
        current_start = block_start + i
        current_offsets = current_start + offsets * inc
        
        mask = current_offsets < (n - 1)
        
        # Read from copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets + inc, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = a_vals + b_vals
        
        # Write to original a array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of elements to process
    num_elements = (n - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE
    )