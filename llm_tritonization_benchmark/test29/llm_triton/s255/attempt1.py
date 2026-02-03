import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < N
    
    # Load b values for current block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # For each element, determine x and y values based on the pattern
    # x = b[i-1] for i > 0, b[N-1] for i = 0
    # y = b[i-2] for i > 1, b[N-2] for i <= 1
    
    x_vals = tl.zeros_like(b_vals)
    y_vals = tl.zeros_like(b_vals)
    
    # Load initial values
    b_last = tl.load(b_ptr + N - 1)
    b_second_last = tl.load(b_ptr + N - 2)
    
    # Handle x values
    # For i = 0: x = b[N-1]
    # For i > 0: x = b[i-1]
    x_vals = tl.where(current_offsets == 0, b_last, 
                      tl.load(b_ptr + (current_offsets - 1), 
                             mask=(current_offsets > 0) & mask, other=0.0))
    
    # Handle y values  
    # For i = 0: y = b[N-2]
    # For i = 1: y = b[N-1] (which is the initial x)
    # For i > 1: y = b[i-2]
    y_vals = tl.where(current_offsets == 0, b_second_last,
                      tl.where(current_offsets == 1, b_last,
                               tl.load(b_ptr + (current_offsets - 2),
                                      mask=(current_offsets > 1) & mask, other=0.0)))
    
    # Compute result: a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x_vals + y_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s255_triton(a, b, x):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    s255_kernel[(grid_size,)](
        a, b, N, BLOCK_SIZE=BLOCK_SIZE
    )