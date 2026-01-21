import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the sequential accumulation pattern
    # Since s depends on previous iterations, we process sequentially
    
    pid = tl.program_id(0)
    
    # Only process with first program to maintain sequential dependency
    if pid != 0:
        return
    
    # Process all elements sequentially
    s = 0.0
    
    # Process in blocks for memory efficiency
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute s values for this block
        s_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for i in range(BLOCK_SIZE):
            if block_start + i < N:
                s += 2.0
                s_vals = tl.where(offsets == i, s, s_vals)
        
        # Compute a[i] = s * b[i] for valid elements
        a_vals = s_vals * b_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program to maintain sequential dependency
    grid = (1,)
    
    s453_kernel[grid](
        a, b, N, BLOCK_SIZE=BLOCK_SIZE
    )