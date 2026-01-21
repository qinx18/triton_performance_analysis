import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets < n - 1) & ((current_offsets % inc) == 0)
        
        valid_offsets = tl.where(mask, current_offsets, 0)
        read_offsets = valid_offsets + inc
        read_mask = mask & (read_offsets < n)
        
        a_vals = tl.load(a_copy_ptr + valid_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + valid_offsets, mask=mask, other=0.0)
        a_read_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        
        result = a_read_vals + b_vals
        
        tl.store(a_ptr + valid_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, n, inc, BLOCK_SIZE=BLOCK_SIZE
    )