import torch
import triton
import triton.language as tl

@triton.jit
def s491_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for indirect assignment: a[ip[i]] = b[i] + c[i] * d[i]
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input arrays with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    ip_vals = tl.load(ip_ptr + offsets, mask=mask)
    
    # Compute values to assign: b[i] + c[i] * d[i]
    values = b_vals + c_vals * d_vals
    
    # Store values at indirect positions: a[ip[i]] = values
    # Each thread handles one indirect store
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            idx = tl.load(ip_ptr + block_start + i)
            val = tl.load(b_ptr + block_start + i) + tl.load(c_ptr + block_start + i) * tl.load(d_ptr + block_start + i)
            tl.store(a_ptr + idx, val)

def s491_triton(a, b, c, d, ip):
    """
    Triton implementation of TSVC s491 - indirect assignment.
    Optimized for GPU with coalesced memory access where possible.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()
    
    n_elements = b.numel()
    
    # Choose block size for good occupancy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s491_kernel[grid](
        a, b, c, d, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a