import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(block_start, n - 1, inc):
        i_offsets = i_start + offsets * inc
        
        # Mask for valid indices
        mask_i = i_offsets < (n - 1)
        mask_i_inc = (i_offsets + inc) < n
        mask = mask_i & mask_i_inc
        
        # Load data
        a_val = tl.load(a_copy_ptr + i_offsets + inc, mask=mask)
        b_val = tl.load(b_ptr + i_offsets, mask=mask)
        
        # Compute and store
        result = a_val + b_val
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )