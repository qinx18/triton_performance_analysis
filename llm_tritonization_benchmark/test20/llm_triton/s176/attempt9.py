import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Create index matrices for convolution
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    
    # Build convolution matrix B where B[i,j] = b[i+m-j-1]
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    B = b[b_indices]  # (m, m)
    
    # Perform matrix multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offset = pid * BLOCK_SIZE
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Load b[i+m-j-1] for current j
        b_indices = i_offsets + m - j - 1
        b_vals = tl.load(b_ptr + b_indices, mask=i_mask, other=0.0)
        
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_val
    
    # Store final result
    tl.store(a_ptr + i_offsets, a_vals, mask=i_mask)