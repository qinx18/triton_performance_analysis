import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr, s,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for bounds checking
    mask = idx < N
    
    # Load indices from ip array
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load values from b array using gathered indices
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Load current values from a array
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute: a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result back to a array
    tl.store(a_ptr + idx, result, mask=mask)

def s4112_triton(a, b, ip, s):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s4112_kernel[grid](
        a, b, ip, s,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )