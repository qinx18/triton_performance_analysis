import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load indices from ip array
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Gather from array b using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute partial dot product
    products = a_vals * b_vals
    
    # Store partial results for reduction
    tl.store(result_ptr + indices, products, mask=mask)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array to store partial results
    partial_results = torch.zeros_like(a)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, partial_results, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final sum
    return torch.sum(partial_results)