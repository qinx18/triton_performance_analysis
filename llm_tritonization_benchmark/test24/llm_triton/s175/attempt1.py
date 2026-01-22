import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n - 1, inc):
        if i + block_start < n - 1 and (i % inc) == 0:
            current_offsets = block_start + offsets
            mask = current_offsets < n - 1
            
            # Check if current offsets match the strided pattern
            stride_mask = ((current_offsets - i) % inc) == 0
            combined_mask = mask & stride_mask & (current_offsets >= i)
            
            if tl.sum(combined_mask.to(tl.int32)) > 0:
                # Load from read-only copy for a[i + inc]
                a_read_offsets = current_offsets + inc
                a_read_mask = combined_mask & (a_read_offsets < n)
                a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
                
                # Load b[i]
                b_vals = tl.load(b_ptr + current_offsets, mask=combined_mask, other=0.0)
                
                # Compute result
                result = a_vals + b_vals
                
                # Store to original array a[i]
                tl.store(a_ptr + current_offsets, result, mask=combined_mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of elements that will be processed
    num_elements = ((n - 1) + inc - 1) // inc
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE=BLOCK_SIZE
    )