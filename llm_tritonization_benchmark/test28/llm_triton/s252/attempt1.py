import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute s = b[i] * c[i] (direct expansion)
    s_vals = b_vals * c_vals
    
    # Compute t values using scalar expansion pattern
    # t = 0 for i=0, t = b[i-1] * c[i-1] for i>0
    
    # Load previous b and c values for t computation
    prev_offsets = block_start + offsets - 1
    prev_mask = mask & ((block_start + offsets) > 0)
    
    prev_b = tl.load(b_ptr + prev_offsets, mask=prev_mask, other=0.0)
    prev_c = tl.load(c_ptr + prev_offsets, mask=prev_mask, other=0.0)
    
    # t = b[i-1] * c[i-1] for i>0, 0 for i=0
    t_vals = tl.where((block_start + offsets) > 0, prev_b * prev_c, 0.0)
    
    # Compute a[i] = s + t
    a_vals = s_vals + t_vals
    
    # Store result
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )