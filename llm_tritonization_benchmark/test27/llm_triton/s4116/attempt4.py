import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, len_2d, inc, j, n_elements, sum_ptr, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < n_elements
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute off = inc + i (scalar expansion applied)
    off_vals = inc + indices
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: (j-1) * len_2d + ip[i]
    aa_indices = (j - 1) * len_2d + ip_vals
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Zero out invalid products
    products = tl.where(mask, products, 0.0)
    
    # Sum all products within this block
    result = tl.sum(products)
    
    # Store result using atomic add
    tl.atomic_add(sum_ptr, result)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n_elements = len_2d - 1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create a tensor to store the sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip, len_2d, inc, j, n_elements, sum_tensor, BLOCK_SIZE
    )
    
    # Return the result
    return sum_tensor.item()