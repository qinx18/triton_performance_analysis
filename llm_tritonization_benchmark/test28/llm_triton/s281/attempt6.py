import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n):
    # Single thread processes all elements sequentially
    x_val = 0.0
    for i in range(n):
        # Read from reverse index for a
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # Compute x value
        x_val = a_val + b_val * c_val
        
        # Store expanded x value
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < threshold) & (indices < n)
    
    # Load x values from expanded array
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    # Store a[i] = x - 1.0
    a_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices >= threshold) & (indices < n)
    
    # Load x values from expanded array
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    # Store a[i] = x - 1.0
    a_vals = x_vals - 1.0
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Create expanded array for x values
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 0: Expand scalar x to array
    s281_expand_x_kernel[(1,)](a, b, c, x_expanded, n)
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: Process first half
    grid_size1 = triton.cdiv(threshold, BLOCK_SIZE)
    if grid_size1 > 0:
        s281_phase1_kernel[(grid_size1,)](a, a_copy, b, c, x_expanded, n, threshold, BLOCK_SIZE)
    
    # Phase 2: Process second half
    remaining = n - threshold
    grid_size2 = triton.cdiv(remaining, BLOCK_SIZE)
    if grid_size2 > 0:
        s281_phase2_kernel[(grid_size2,)](a, b, c, x_expanded, n, threshold, BLOCK_SIZE)