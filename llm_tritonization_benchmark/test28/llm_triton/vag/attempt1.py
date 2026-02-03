import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices from ip array
        indices = tl.load(ip_ptr + current_offsets, mask=mask)
        
        # Gather from b array using indices
        gathered = tl.load(b_ptr + indices, mask=mask)
        
        # Store to a array
        tl.store(a_ptr + current_offsets, gathered, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    vag_kernel[(1,)](
        a, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )