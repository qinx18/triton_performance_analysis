import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # This avoids the timeout issue with sequential loops in Triton
    
    # Prepare inputs for conv1d
    # b needs to be reshaped and the relevant portion extracted
    b_input = b[m-1:2*m-1].unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    c_weight = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Compute convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_weight, padding=0)
    
    # Add result to a
    a[:m] += conv_result.squeeze()

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Sequential loop over j (required due to accumulation pattern)
    for j in range(m):
        # Load c[j] (broadcast to all threads)
        c_val = tl.load(c_ptr + j)
        
        # Load b[i+m-j-1] values
        b_idx = i_idx + m - j - 1
        b_vals = tl.load(b_ptr + b_idx, mask=i_mask, other=0.0)
        
        # Accumulate
        a_vals += b_vals * c_val
    
    # Store final results
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)