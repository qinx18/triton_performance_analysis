import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < N
    
    # Load a values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load indirect indices
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products to output for reduction
    tl.store(output_ptr + indices, products, mask=mask)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    
    # Allocate output tensor for partial products
    output = torch.zeros_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, output,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    return output.sum().item()