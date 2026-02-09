import triton
import triton.language as tl
import torch

@triton.jit
def deriche_kernel(imgIn_ptr, imgOut_ptr, y2_ptr, yy1_ptr, alpha, H, W, 
                   imgIn_stride_0, imgIn_stride_1,
                   imgOut_stride_0, imgOut_stride_1,
                   y2_stride_0, y2_stride_1,
                   yy1_stride_0, yy1_stride_1,
                   BLOCK_SIZE: tl.constexpr):
    
    # Calculate coefficients
    exp_alpha = tl.exp(-alpha)
    exp_2alpha = tl.exp(-2.0 * alpha)
    
    k = (1.0 - exp_alpha) * (1.0 - exp_alpha) / (1.0 + 2.0 * alpha * exp_alpha - exp_2alpha)
    a1 = a5 = k
    a2 = a6 = k * exp_alpha * (alpha - 1.0)
    a3 = a7 = k * exp_alpha * (alpha + 1.0)
    a4 = a8 = -k * exp_2alpha
    b1 = tl.exp2(-alpha)
    b2 = -exp_2alpha
    c1 = c2 = 1.0
    
    # First pass: forward direction over j for each i
    for i in range(W):
        ym1 = 0.0
        ym2 = 0.0
        xm1 = 0.0
        
        for j in range(H):
            imgIn_val = tl.load(imgIn_ptr + i * imgIn_stride_0 + j * imgIn_stride_1)
            yy1_val = a1 * imgIn_val + a2 * xm1 + b1 * ym1 + b2 * ym2
            tl.store(yy1_ptr + i * yy1_stride_0 + j * yy1_stride_1, yy1_val)
            
            xm1 = imgIn_val
            ym2 = ym1
            ym1 = yy1_val
    
    # Second pass: backward direction over j for each i
    for i in range(W):
        yp1 = 0.0
        yp2 = 0.0
        xp1 = 0.0
        xp2 = 0.0
        
        for j in range(H-1, -1, -1):
            y2_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + i * y2_stride_0 + j * y2_stride_1, y2_val)
            
            imgIn_val = tl.load(imgIn_ptr + i * imgIn_stride_0 + j * imgIn_stride_1)
            xp2 = xp1
            xp1 = imgIn_val
            yp2 = yp1
            yp1 = y2_val
    
    # Third pass: combine yy1 and y2 into imgOut
    for i in range(W):
        for j in range(H):
            yy1_val = tl.load(yy1_ptr + i * yy1_stride_0 + j * yy1_stride_1)
            y2_val = tl.load(y2_ptr + i * y2_stride_0 + j * y2_stride_1)
            imgOut_val = c1 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + i * imgOut_stride_0 + j * imgOut_stride_1, imgOut_val)
    
    # Fourth pass: forward direction over i for each j
    for j in range(H):
        tm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        
        for i in range(W):
            imgOut_val = tl.load(imgOut_ptr + i * imgOut_stride_0 + j * imgOut_stride_1)
            yy1_val = a5 * imgOut_val + a6 * tm1 + b1 * ym1 + b2 * ym2
            tl.store(yy1_ptr + i * yy1_stride_0 + j * yy1_stride_1, yy1_val)
            
            tm1 = imgOut_val
            ym2 = ym1
            ym1 = yy1_val
    
    # Fifth pass: backward direction over i for each j
    for j in range(H):
        tp1 = 0.0
        tp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        
        for i in range(W-1, -1, -1):
            y2_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + i * y2_stride_0 + j * y2_stride_1, y2_val)
            
            imgOut_val = tl.load(imgOut_ptr + i * imgOut_stride_0 + j * imgOut_stride_1)
            tp2 = tp1
            tp1 = imgOut_val
            yp2 = yp1
            yp1 = y2_val
    
    # Sixth pass: final combination
    for i in range(W):
        for j in range(H):
            yy1_val = tl.load(yy1_ptr + i * yy1_stride_0 + j * yy1_stride_1)
            y2_val = tl.load(y2_ptr + i * y2_stride_0 + j * y2_stride_1)
            imgOut_val = c2 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + i * imgOut_stride_0 + j * imgOut_stride_1, imgOut_val)

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 128
    
    grid = (1,)
    
    deriche_kernel[grid](
        imgIn, imgOut, y2, yy1, alpha, H, W,
        imgIn.stride(0), imgIn.stride(1),
        imgOut.stride(0), imgOut.stride(1),
        y2.stride(0), y2.stride(1),
        yy1.stride(0), yy1.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )