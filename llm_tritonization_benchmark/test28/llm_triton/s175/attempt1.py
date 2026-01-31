import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(0, n, BLOCK_SIZE):
        current_offsets = start + offsets
        
        # Check bounds and ensure stride access is valid
        read_offsets = current_offsets + inc
        mask = (current_offsets < n) & (read_offsets < (n + inc))
        
        # Load values
        a_vals = tl.load(a_copy_ptr + read_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute result
        result = a_vals + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE=BLOCK_SIZE
    )