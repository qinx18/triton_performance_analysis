import torch
import triton
import triton.language as tl

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices and values
        indices = tl.load(ip_ptr + current_offsets, mask=mask)
        values = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Scatter: a[ip[i]] = b[i] for each valid element
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                idx = tl.load(ip_ptr + block_start + i)
                val = tl.load(b_ptr + block_start + i)
                tl.store(a_ptr + idx, val)

def vas_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (1,)
    
    vas_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )