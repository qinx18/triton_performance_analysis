import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, output_ptr, len_2d, inc, j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < len_2d - 1
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Compute off = inc + i
    off_vals = inc + indices
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask)
    
    # Compute aa indices: (j-1) * len_2d + ip[i]
    aa_indices = (j - 1) * len_2d + ip_vals
    
    # Load aa[j-1][ip[i]] values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
    
    # Compute products
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Store products for reduction
    tl.store(output_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    BLOCK_SIZE = 256
    n_elements = len_2d - 1
    
    # Create output tensor for partial products
    output = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip, output, len_2d, inc, j, BLOCK_SIZE
    )
    
    # Sum all products to get final result
    return output.sum().item()