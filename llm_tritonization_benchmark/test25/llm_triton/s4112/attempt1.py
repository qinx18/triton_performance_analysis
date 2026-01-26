import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr, s,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Gather from b array using ip indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s4112_triton(a, b, ip, s):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )