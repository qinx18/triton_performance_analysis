import torch

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Reshape for conv1d: (batch, channels, length)
    b_input = b.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=m-1)
    
    # Extract the relevant portion and add to a
    a[:m] += conv_result[0, 0, m-1:2*m-1]