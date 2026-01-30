import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices (i < n_elements-1)
    mask = indices < n_elements - 1
    
    # Load b[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Load a[j] where j = i + 1, so a[i+1]
    a_vals = tl.load(a_ptr + indices + 1, mask=mask, other=0.0)
    
    # Compute a[i] = a[j] + b[i] = a[i+1] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0]
    
    # We process n_elements-1 elements (i from 0 to LEN_1D-2)
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )