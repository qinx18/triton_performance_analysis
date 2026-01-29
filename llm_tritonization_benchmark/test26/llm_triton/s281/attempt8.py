import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(x_expanded, a, b, c, N, threshold):
    # Single thread processes all elements sequentially
    x_val = 0.0
    for i in range(N):
        if i < threshold:
            # Read from original values (high indices)
            a_val = tl.load(a + (N - 1 - i))
        else:
            # Read from updated values (low indices)
            a_val = tl.load(a + (N - 1 - i))
        
        b_val = tl.load(b + i)
        c_val = tl.load(c + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded + i, x_val)

@triton.jit
def s281_phase1_kernel(a, b, c, x_expanded, a_original, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load x values from expanded array
    x_vals = tl.load(x_expanded + indices, mask=mask)
    
    # Store results
    tl.store(a + indices, x_vals - 1.0, mask=mask)
    tl.store(b + indices, x_vals, mask=mask)

@triton.jit
def s281_sequential_kernel(a, b, c, a_original, N, threshold):
    # Process all elements sequentially to handle dependencies correctly
    for i in range(N):
        if i < threshold:
            # Read from original a values
            a_val = tl.load(a_original + (N - 1 - i))
        else:
            # Read from updated a values
            a_val = tl.load(a + (N - 1 - i))
        
        b_val = tl.load(b + i)
        c_val = tl.load(c + i)
        x_val = a_val + b_val * c_val
        
        # Update a and b
        tl.store(a + i, x_val - 1.0)
        tl.store(b + i, x_val)

def s281_triton(a, b, c):
    N = a.shape[0]
    threshold = N // 2
    
    # Store original a values
    a_original = a.clone()
    
    # Use sequential processing to handle the crossing threshold correctly
    grid = (1,)
    s281_sequential_kernel[grid](a, b, c, a_original, N, threshold)