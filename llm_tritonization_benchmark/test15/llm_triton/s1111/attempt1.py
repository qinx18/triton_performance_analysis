import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
        result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
        
        # Store to a[2*i] positions
        output_offsets = 2 * current_offsets
        output_mask = current_offsets < n_elements
        tl.store(a_ptr + output_offsets, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    s1111_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )