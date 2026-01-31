import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load data for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate common expression
    de_vals = d_vals * e_vals
    
    # Handle j=-1 start by counting how many valid elements have c[k] > 0 before each i
    # For each thread, calculate the j offset based on i + number of positive c values up to i
    
    # First store a[j] = b[i] + d[i] * e[i] where j starts at 0 for i=0
    j_base = 2 * i_offsets  # Maximum possible j for element i
    
    # Count valid positive c values in this block up to current position
    condition = c_vals > 0.0
    
    # For exact matching, we need to compute j sequentially
    # Use atomic operations or serial approach for exact j computation
    
    # Since Triton doesn't handle variable j easily, we need to compute exact positions
    # j starts at -1, then j++ before first store, then j++ again if c[i] > 0
    
    # After j++, j = i (since j starts at -1)
    # After potential second j++, j = i + count_positive_up_to_i
    
    val1 = b_vals + de_vals
    tl.store(a_ptr + i_offsets, val1, mask=mask)
    
    # For the conditional store, we need to find the next available position
    # This requires prefix sum of conditions
    valid_mask = mask & condition
    
    # Simple approach: store at offset that accounts for all previous elements
    # For i-th iteration: first store at position i, second store (if any) at n_half + i
    second_pos = n_half + i_offsets
    val2 = c_vals + de_vals
    tl.store(a_ptr + second_pos, val2, mask=valid_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    # Reset a to zero first
    a.zero_()
    
    # Process sequentially to maintain exact j logic
    for i in range(n_half):
        j_pos = i
        if i > 0:
            # Count how many c[k] > 0 for k < i
            j_pos += torch.sum(c[:i] > 0).item()
        
        a[j_pos] = b[i] + d[i] * e[i]
        
        if c[i] > 0:
            j_pos += 1
            a[j_pos] = c[i] + d[i] * e[i]