import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Gather from b array using loaded indices
    gathered_values = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Store to a array
    tl.store(a_ptr + indices, gathered_values, mask=mask)

def vag_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    vag_kernel[(num_blocks,)](
        a, b, ip, n_elements, BLOCK_SIZE
    )