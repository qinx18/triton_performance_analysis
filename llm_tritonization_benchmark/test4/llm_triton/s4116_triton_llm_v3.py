import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, inc, j, n, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid indices
    mask = i_offsets < n
    
    # Initialize local sum
    local_sum = 0.0
    
    # Compute off = inc + i
    off_indices = inc + i_offsets
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_indices, mask=mask, other=0.0)
    
    # Load ip[i] values for indirect addressing
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Compute aa indices: (j-1) * LEN_2D + ip[i]
    aa_indices = (j - 1) * LEN_2D + ip_vals
    
    # Load aa[j-1][ip[i]] values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * aa_vals
    local_sum = tl.sum(tl.where(mask, products, 0.0))
    
    # Atomic add to global sum
    tl.atomic_add(sum_ptr, local_sum)

def s4116_triton(a, aa, ip, inc, j, LEN_2D):
    n = LEN_2D - 1
    
    # Initialize sum tensor
    sum_tensor = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    s4116_kernel[(num_blocks,)](
        a, aa, ip, sum_tensor, inc, j, n, LEN_2D, BLOCK_SIZE
    )
    
    return sum_tensor.item()