import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[LEN_1D/2] (scalar broadcast)
    mid_idx = n_elements // 2
    a_mid = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[LEN_1D/2] + b[i]
    result = a_mid + b_vals
    
    # Store to original a array
    tl.store(a_ptr + idx, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE,
    )