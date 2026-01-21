import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load indices from ip array and ensure they're valid
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Gather from array b using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products for reduction
    tl.store(output_ptr + indices, products, mask=mask)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output array for partial products
    output = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, output, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    return torch.sum(output)