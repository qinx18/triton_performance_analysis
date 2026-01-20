import torch
import triton
import triton.language as tl

@triton.jit
def s254_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Load x value (b[N-1])
    x = tl.load(b_ptr + N - 1)
    
    # Compute a[i] = (b[i] + x) * 0.5
    a_vals = (b_vals + x) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s254_update_x_kernel(b_ptr, x_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the x = b[i] update by creating an expanded array
    # Since this creates a dependency, we process sequentially
    if tl.program_id(0) == 0:  # Only first program processes
        for i in range(N):
            if i == 0:
                x_val = tl.load(b_ptr + N - 1)  # Initial x = b[LEN_1D-1]
            else:
                x_val = tl.load(b_ptr + i - 1)  # x = b[i-1] for subsequent iterations
            tl.store(x_array_ptr + i, x_val)

@triton.jit
def s254_final_kernel(a_ptr, b_ptr, x_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_array_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = (b[i] + x) * 0.5
    a_vals = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded x values
    x_array = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand x values
    grid = (1,)  # Single program for sequential processing
    s254_update_x_kernel[grid](b, x_array, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation using expanded x
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_final_kernel[grid](a, b, x_array, N, BLOCK_SIZE=BLOCK_SIZE)