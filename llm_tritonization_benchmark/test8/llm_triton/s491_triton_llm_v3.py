import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i], c[i], d[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load ip[i] values (indices for scatter)
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask)
        
        # Compute b[i] + c[i] * d[i]
        result = b_vals + c_vals * d_vals
        
        # Scatter to a[ip[i]] - need to handle each element individually
        # Since we can't vectorize scatter with indirect addressing,
        # we need to store each element separately
        for i in range(BLOCK_SIZE):
            element_mask = mask & (offsets == i)
            if tl.sum(element_mask.to(tl.int32)) > 0:
                idx = tl.load(ip_ptr + block_start + i)
                val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
                tl.store(a_ptr + idx, val)

def s491_triton(a, b, c, d, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s491_kernel[grid](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )