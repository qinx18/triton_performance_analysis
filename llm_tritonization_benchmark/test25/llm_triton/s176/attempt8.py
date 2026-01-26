import triton
import triton.language as tl
import torch

def s176_triton(a, b, c, m):
    # Use PyTorch's conv1d for efficient convolution computation
    # The pattern a[i] += b[i+m-j-1] * c[j] is a convolution
    
    # Prepare tensors for conv1d
    # conv1d expects (batch_size, channels, length)
    b_input = b.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, len(b))
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, m)
    
    # Perform convolution
    # We need to extract the right portion of the convolution result
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=0)
    
    # The convolution result starts at index m-1, we want indices m-1 to 2*m-2
    # But we need to map this correctly to a[0:m]
    start_idx = m - 1
    a[:m] += conv_result[0, 0, start_idx:start_idx + m]