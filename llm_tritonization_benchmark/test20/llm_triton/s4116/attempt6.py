import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, inc, j, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < (len_2d - 1)
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute aa indices: (j-1) * len_2d + ip[i]
    aa_indices = (j - 1) * len_2d + ip_vals
    
    # Compute a indices: inc + i
    a_indices = inc + indices
    
    # Load values
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Zero out invalid elements
    products = tl.where(mask, products, 0.0)
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Store block sum
    tl.atomic_add(sum_ptr, block_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Initialize sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    
    s4116_kernel[grid](
        a, aa.view(-1), ip, sum_result, 
        inc, j, len_2d, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()