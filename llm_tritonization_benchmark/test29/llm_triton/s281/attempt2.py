import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < (n // 2)
    
    reverse_offsets = n - 1 - i_offsets
    
    a_vals = tl.load(a_copy_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + i_offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + i_offsets, x, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + (n // 2)
    
    mask = i_offsets < n
    
    reverse_offsets = n - 1 - i_offsets
    
    a_vals = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + i_offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + i_offsets, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1
    grid_size_phase1 = triton.cdiv(threshold, BLOCK_SIZE)
    if grid_size_phase1 > 0:
        s281_kernel_phase1[(grid_size_phase1,)](
            a, a_copy, b, c, n, BLOCK_SIZE
        )
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    grid_size_phase2 = triton.cdiv(remaining, BLOCK_SIZE)
    if grid_size_phase2 > 0:
        s281_kernel_phase2[(grid_size_phase2,)](
            a, b, c, n, BLOCK_SIZE
        )