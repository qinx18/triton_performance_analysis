import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Convert to reverse order indices
    reverse_indices = n_elements - 1 - (block_start + offsets)
    
    mask = (block_start + offsets) < n_elements
    reverse_mask = reverse_indices >= 0
    final_mask = mask & reverse_mask
    
    # Load b values using reverse indices
    b_vals = tl.load(b_ptr + reverse_indices, mask=final_mask, other=0.0)
    
    # Compute a[i] = b[i] + 1.0
    a_vals = b_vals + 1.0
    
    # Store to a using reverse indices
    tl.store(a_ptr + reverse_indices, a_vals, mask=final_mask)

def s1112_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )