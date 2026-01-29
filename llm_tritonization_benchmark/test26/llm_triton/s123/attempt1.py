import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_input, BLOCK_SIZE: tl.constexpr):
    # Process in blocks along the input dimension
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = block_start + offsets
    input_mask = input_offsets < n_input
    
    # Load input data
    b_vals = tl.load(b_ptr + input_offsets, mask=input_mask, other=0.0)
    c_vals = tl.load(c_ptr + input_offsets, mask=input_mask, other=0.0)
    d_vals = tl.load(d_ptr + input_offsets, mask=input_mask, other=0.0)
    e_vals = tl.load(e_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Compute first output values (always written)
    out1_vals = b_vals + d_vals * e_vals
    
    # Compute second output values (conditionally written)
    out2_vals = c_vals + d_vals * e_vals
    
    # Determine which elements have c > 0
    c_positive = c_vals > 0.0
    
    # Calculate output positions
    # First output goes to j = 2*i (since j starts at -1 and increments to 0, 1, 2, ...)
    # Second output goes to j = 2*i + 1 when condition is met
    out1_positions = 2 * input_offsets
    out2_positions = 2 * input_offsets + 1
    
    # Determine valid outputs
    valid_out1 = input_mask
    valid_out2 = input_mask & c_positive
    
    # Store first outputs
    tl.store(a_ptr + out1_positions, out1_vals, mask=valid_out1)
    
    # Store second outputs where condition is met
    tl.store(a_ptr + out2_positions, out2_vals, mask=valid_out2)

def s123_triton(a, b, c, d, e):
    n_input = b.shape[0] // 2  # Loop goes to LEN_1D/2
    
    # Clear the output array first
    a.zero_()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_input, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_input,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a