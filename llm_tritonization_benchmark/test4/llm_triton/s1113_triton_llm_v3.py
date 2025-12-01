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
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load a[LEN_1D/2] once per block (scalar broadcast)
    mid_idx = n_elements // 2
    a_mid = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] for this block
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[LEN_1D/2] + b[i]
    result = a_mid + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

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
        BLOCK_SIZE=BLOCK_SIZE,
    )