import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Pad b to handle the indexing pattern b[i+m-j-1]
    b_padded = torch.cat([torch.zeros(m-1, device=b.device, dtype=b.dtype), b])
    
    # Create convolution kernel from c (reverse order for convolution)
    c_kernel = torch.flip(c[:m], [0])
    
    # Reshape for conv1d: (batch_size, channels, length)
    b_input = b_padded.unsqueeze(0).unsqueeze(0)
    c_weight = c_kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_weight, padding=0)
    
    # Extract the relevant portion and add to a
    a[:m] += conv_result[0, 0, :m]