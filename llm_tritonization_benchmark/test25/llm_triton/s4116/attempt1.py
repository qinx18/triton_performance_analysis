import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr, sum_ptr,
    len_2d_minus_1, aa_stride_0, aa_stride_1,
    inc, j_minus_1,
    BLOCK_SIZE: tl.constexpr
):
    # Get block information
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < len_2d_minus_1
    
    # Initialize partial sum
    partial_sum = 0.0
    
    # Load indices for this block
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute offsets for array a
    a_offsets = inc + indices
    
    # Load from array a
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Compute aa offsets: aa[j-1][ip[i]]
    aa_offsets = j_minus_1 * aa_stride_0 + ip_indices * aa_stride_1
    
    # Load from array aa
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * aa_vals
    partial_sum = tl.sum(tl.where(mask, products, 0.0))
    
    # Store partial sum
    tl.atomic_add(sum_ptr, partial_sum)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    # Derive dimensions from input shapes
    len_2d_minus_1 = len_2d - 1
    j_minus_1 = j - 1
    
    # Create output tensor for sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Configure grid and block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(len_2d_minus_1, BLOCK_SIZE)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip, sum_tensor,
        len_2d_minus_1, aa.stride(0), aa.stride(1),
        inc, j_minus_1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_tensor.item()