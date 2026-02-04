import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    d_vals = tl.load(d_ptr + indices, mask=mask, other=1.0)
    has_negative = tl.sum(tl.where(d_vals < 0.0, 1, 0)) > 0
    
    if not has_negative:
        a_vals = tl.load(a_ptr + indices, mask=mask)
        b_vals = tl.load(b_ptr + indices, mask=mask)
        c_vals = tl.load(c_ptr + indices, mask=mask)
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + indices, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Check for early exit condition
    negative_mask = d < 0.0
    if torch.any(negative_mask):
        # Find first negative index
        exit_idx = torch.argmax(negative_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s481_kernel[(num_blocks,)](
        a, b, c, d, n_elements, BLOCK_SIZE
    )