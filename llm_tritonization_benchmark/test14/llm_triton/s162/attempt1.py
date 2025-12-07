import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    c_ptr,
    k,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n_elements) & (k > 0)
    
    # Load from read-only copy with forward offset
    a_read_indices = indices + k
    a_read_mask = mask & (a_read_indices < (n_elements + k))
    a_vals = tl.load(a_copy_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Load b and c arrays
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute: a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a,
        a_copy,
        b,
        c,
        k,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )