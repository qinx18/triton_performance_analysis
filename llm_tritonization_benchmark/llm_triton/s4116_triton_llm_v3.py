import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr,
    j, inc, len_2d,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the range of indices this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = offsets < (len_2d - 1)
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Calculate offsets for array a
    a_offsets = inc + offsets
    
    # Load from array a
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate 2D indices for aa array
    row_idx = j - 1
    aa_offsets = row_idx * len_2d + ip_indices
    
    # Load from aa array
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Sum the products
    partial_sum = tl.sum(products)
    
    # Store partial sum (will be reduced on host)
    tl.store(tl.program_id(0) + tl.zeros([1], dtype=tl.float32), partial_sum)

def s4116_triton(a, aa, ip, j, inc, len_2d):
    # Ensure inputs are on GPU
    device = a.device
    aa = aa.to(device)
    ip = ip.to(device)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    num_elements = len_2d - 1
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=torch.float32, device=device)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip,
        j, inc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reduce partial sums on host
    total_sum = partial_sums.sum().item()
    
    return total_sum