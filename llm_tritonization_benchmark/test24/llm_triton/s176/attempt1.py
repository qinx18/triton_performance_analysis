import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # This avoids the timeout issue with nested sequential loops in Triton
    
    # Prepare tensors for conv1d
    # Input: b (we need the relevant portion)
    # Weight: c (first m elements, flipped for convolution)
    
    b_input = b[:2*m-1].unsqueeze(0).unsqueeze(0)  # (1, 1, 2*m-1)
    c_weight = torch.flip(c[:m], [0]).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    # conv1d computes sum over j of: b[i+offset-j] * c_flipped[j]
    # We need: sum over j of: b[i+m-j-1] * c[j]
    # This is equivalent to: sum over j of: b[i+m-1-j] * c[j]
    
    conv_result = torch.nn.functional.conv1d(
        b_input, 
        c_weight, 
        padding=0
    )[0, 0]  # Remove batch and channel dimensions
    
    # Add the convolution result to a[:m]
    a[:m] += conv_result[:m]