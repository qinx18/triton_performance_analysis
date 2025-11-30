import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(
    a_ptr, b_ptr, c_ptr,
    LEN_1D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    m = LEN_1D // 2
    
    # Get program ID for parallelization across j dimension
    j = tl.program_id(0)
    
    if j >= m:
        return
    
    # Load c[j] once for this thread block
    c_j = tl.load(c_ptr + j)
    
    # Process all i values for this j
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < m
    
    # Calculate b index: i + m - j - 1
    b_indices = i_offsets + m - j - 1
    
    # Load current a values
    a_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Load b values
    b_vals = tl.load(b_ptr + b_indices, mask=i_mask, other=0.0)
    
    # Compute update: a[i] += b[i+m-j-1] * c[j]
    update = b_vals * c_j
    new_a_vals = a_vals + update
    
    # Store back to a
    tl.store(a_ptr + i_offsets, new_a_vals, mask=i_mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    m = LEN_1D // 2
    
    BLOCK_SIZE = triton.next_power_of_2(m)
    
    # Sequential execution over j dimension to handle dependencies
    for j in range(m):
        grid = (1,)
        s176_kernel[grid](
            a, b, c,
            LEN_1D=LEN_1D,
            BLOCK_SIZE=BLOCK_SIZE,
        )