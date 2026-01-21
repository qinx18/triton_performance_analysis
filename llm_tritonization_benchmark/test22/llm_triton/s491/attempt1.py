import triton
import triton.language as tl
import torch

@triton.jit
def s491_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load b, c, d arrays
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute b[i] + c[i] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Scatter to a[ip[i]]
    # Use atomic store to handle potential conflicts
    tl.atomic_add(a_ptr + ip_indices, result, mask=mask)

def s491_triton(a, b, c, d, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Zero out the output array first since we're using atomic_add
    a.zero_()
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s491_kernel[grid](
        a, b, c, d, ip, N, BLOCK_SIZE=BLOCK_SIZE
    )