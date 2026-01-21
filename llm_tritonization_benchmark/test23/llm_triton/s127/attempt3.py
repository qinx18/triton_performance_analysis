import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = block_start + offsets
    
    # Mask for valid input indices (i < n//2)
    input_mask = input_offsets < n // 2
    
    # Load input values with masking
    b_vals = tl.load(b_ptr + input_offsets, mask=input_mask, other=0.0)
    c_vals = tl.load(c_ptr + input_offsets, mask=input_mask, other=0.0)
    d_vals = tl.load(d_ptr + input_offsets, mask=input_mask, other=0.0)
    e_vals = tl.load(e_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Calculate output indices (j values)
    j1_offsets = 2 * input_offsets      # j++ first time
    j2_offsets = 2 * input_offsets + 1  # j++ second time
    
    # Calculate results
    result1 = b_vals + c_vals * d_vals
    result2 = b_vals + d_vals * e_vals
    
    # Store results with appropriate masks
    j1_mask = input_mask & (j1_offsets < n)
    j2_mask = input_mask & (j2_offsets < n)
    
    tl.store(a_ptr + j1_offsets, result1, mask=j1_mask)
    tl.store(a_ptr + j2_offsets, result2, mask=j2_mask)

def s127_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    
    s127_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)