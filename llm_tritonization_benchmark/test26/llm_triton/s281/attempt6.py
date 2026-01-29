import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a, b, c, x_expanded, N):
    # Single thread processes all elements sequentially to compute expanded x values
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(N):
        # Compute x = a[N-1-i] + b[i] * c[i]
        a_val = tl.load(a + (N - 1 - i))
        b_val = tl.load(b + i)
        c_val = tl.load(c + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded + i, x_val)

@triton.jit
def s281_phase1_kernel(a, a_copy, b, c, x_expanded, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < threshold)
    
    # Load x values
    x_vals = tl.load(x_expanded + indices, mask=mask)
    
    # Store results
    a_vals = x_vals - 1.0
    b_vals = x_vals
    
    tl.store(a + indices, a_vals, mask=mask)
    tl.store(b + indices, b_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a, b, c, x_expanded, threshold, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < N)
    
    # Load x values
    x_vals = tl.load(x_expanded + indices, mask=mask)
    
    # Store results
    a_vals = x_vals - 1.0
    b_vals = x_vals
    
    tl.store(a + indices, a_vals, mask=mask)
    tl.store(b + indices, b_vals, mask=mask)

def s281_triton(a, b, c):
    N = a.shape[0]
    threshold = N // 2
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Step 1: Compute expanded x values sequentially
    s281_expand_x_kernel[(1,)](a_copy, b, c, x_expanded, N)
    
    # Step 2: Phase 1 - parallel update of first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, x_expanded, threshold, BLOCK_SIZE)
    
    # Step 3: Phase 2 - parallel update of second half
    grid2 = (triton.cdiv(N - threshold, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, x_expanded, threshold, N, BLOCK_SIZE)