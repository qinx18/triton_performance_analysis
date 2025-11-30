import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(
    a_ptr, aa_ptr, ip_ptr, result_ptr,
    inc, j_minus_1, len_2d_minus_1, len_2d,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    sum_val = 0.0
    
    for block_start in range(0, len_2d_minus_1, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, len_2d_minus_1)
        block_size = block_end - block_start
        
        i_vals = tl.arange(0, BLOCK_SIZE) + block_start
        mask = i_vals < len_2d_minus_1
        
        off = inc + i_vals
        ip_indices = tl.load(ip_ptr + i_vals, mask=mask, other=0)
        aa_indices = j_minus_1 * len_2d + ip_indices
        
        a_vals = tl.load(a_ptr + off, mask=mask, other=0.0)
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        products = a_vals * aa_vals
        products = tl.where(mask, products, 0.0)
        sum_val += tl.sum(products)
    
    tl.store(result_ptr, sum_val)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[1]
    len_2d_minus_1 = len_2d - 1
    j_minus_1 = j - 1
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_kernel[grid](
        a, aa, ip, result,
        inc, j_minus_1, len_2d_minus_1, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()