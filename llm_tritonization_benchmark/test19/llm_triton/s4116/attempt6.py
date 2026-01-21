import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d, sum_ptr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < len_2d - 1
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate a indices: off = inc + i
    a_indices = inc + indices
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    
    # Calculate aa indices: aa[j-1][ip[i]]
    aa_row = j - 1
    aa_indices = aa_row * len_2d + ip_vals
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    products = tl.where(mask, products, 0.0)
    
    # Store products for this block
    tl.store(sum_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    n = len_2d - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Create temporary array to store partial products
    temp_products = torch.zeros(n, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip, inc, j, len_2d, temp_products, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products
    result = torch.sum(temp_products)
    return result.item()