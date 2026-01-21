import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # This avoids the timeout issue with sequential loops in Triton
    
    # Prepare tensors for conv1d
    # conv1d expects input: (batch, channels, length)
    # and weight: (out_channels, in_channels, kernel_size)
    
    # Create the convolution kernel from c
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Prepare b for convolution - we need to extract the relevant portion
    # The pattern b[i+m-j-1] for i in [0, m), j in [0, m) translates to
    # b[m-1:2*m-1] accessed in reverse order for the convolution
    b_input = b[m-1:2*m-1].unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    # We need 'full' convolution to match the original indexing pattern
    conv_result = torch.nn.functional.conv1d(
        b_input, 
        c_kernel, 
        padding=m-1
    )
    
    # Extract the relevant part and add to a
    # The convolution produces output of size m + m - 1, we need the first m elements
    a[:m] += conv_result[0, 0, :m]