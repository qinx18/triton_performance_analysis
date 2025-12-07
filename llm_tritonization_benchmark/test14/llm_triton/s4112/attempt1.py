import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(block_start, n_elements, tl.num_programs(0) * BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Load indices
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Gather b values using indices
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute and store
        result = a_vals + b_vals * s
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s4112_kernel[(num_programs,)](a, b, ip, s, n_elements, BLOCK_SIZE)