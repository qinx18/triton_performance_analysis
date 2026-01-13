import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr, 
    inc, j,
    n_elements,
    aa_stride_0, aa_stride_1,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = tl.zeros([1], dtype=tl.float32)[0]
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load ip values
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Compute off = inc + i for each element
        off_vals = inc + current_offsets
        
        # Load a[off] values
        a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
        
        # Compute aa indices: aa[j-1][ip[i]]
        row_idx = j - 1
        aa_offsets = row_idx * aa_stride_0 + ip_vals * aa_stride_1
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * aa_vals
        products = tl.where(mask, products, 0.0)
        sum_val += tl.sum(products)
    
    tl.store(tl.arange(0, 1), sum_val)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    
    BLOCK_SIZE = 256
    
    # Get strides for aa array
    aa_stride_0 = aa.stride(0)
    aa_stride_1 = aa.stride(1)
    
    # Create output tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s4116_kernel[1, 1](
        a, aa, ip, result,
        inc, j,
        n_elements,
        aa_stride_0, aa_stride_1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()