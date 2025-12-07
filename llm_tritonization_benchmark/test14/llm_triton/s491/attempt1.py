import triton
import triton.language as tl
import torch

@triton.jit
def s491_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b, c, d values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load indices
        indices = tl.load(ip_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = b_vals + c_vals * d_vals
        
        # Scatter store - handle each element individually
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                idx_offset = tl.load(ip_ptr + block_start + i)
                val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
                tl.store(a_ptr + idx_offset, val)

def s491_triton(a, b, c, d, ip):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    
    s491_kernel[(1,)](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a