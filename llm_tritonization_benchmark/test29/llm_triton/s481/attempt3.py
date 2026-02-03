import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Perform computation: a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Phase 1: Check for exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Exit check happens before computation, so process elements before exit_idx
        n_elements = exit_idx
    
    # Phase 2: Process valid elements with Triton kernel
    if n_elements > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s481_kernel[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )