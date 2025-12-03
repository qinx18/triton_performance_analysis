import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load from read-only copy (i + m where m = 1)
    a_read_offsets = idx + 1
    a_read_mask = mask & (a_read_offsets < (n_elements + 1))
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Load b values
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )