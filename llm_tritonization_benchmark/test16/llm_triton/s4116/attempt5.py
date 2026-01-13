import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, sum_ptr, N):
    BLOCK_SIZE = 256
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Load indices
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + offsets
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: (j-1) * N + ip[i]
    aa_indices = (j - 1) * N + ip_vals
    
    # Load aa[j-1][ip[i]] values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Apply mask to products
    products = tl.where(mask, products, 0.0)
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Atomic add to global sum
    tl.atomic_add(sum_ptr, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    N = aa.shape[0] - 1  # LEN_2D - 1
    
    # Initialize sum tensor
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Calculate grid size
    grid_size = triton.cdiv(N, 256)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip, inc, j, sum_tensor, N
    )
    
    return sum_tensor.item()