import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Gather from b array using the loaded indices
    gathered_values = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Store to a array
    tl.store(a_ptr + indices, gathered_values, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )