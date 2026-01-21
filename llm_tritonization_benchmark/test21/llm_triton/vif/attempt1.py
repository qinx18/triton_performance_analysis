import triton
import triton.language as tl

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    condition = b_vals > 0.0
    
    # Only update a[i] where condition is true
    update_mask = mask & condition
    tl.store(a_ptr + offsets, b_vals, mask=update_mask)

def vif_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)