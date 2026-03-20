import triton
import triton.language as tl
import torch
import math

@triton.jit
def deriche_horizontal_kernel(imgOut, imgIn, yy1, y2,
                             a1, a2, a3, a4, b1, b2, c1, 
                             H: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0)
    
    # Forward pass
    ym1 = 0.0
    ym2 = 0.0
    xm1 = 0.0
    for j in range(H):
        idx = row * H + j
        val = tl.load(imgIn + idx)
        out = a1 * val + a2 * xm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1 + idx, out)
        xm1 = val
        ym2 = ym1
        ym1 = out
    
    # Backward pass
    yp1 = 0.0
    yp2 = 0.0
    xp1 = 0.0
    xp2 = 0.0
    for j in range(H-1, -1, -1):
        idx = row * H + j
        val = tl.load(imgIn + idx)
        out = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
        tl.store(y2 + idx, out)
        xp2 = xp1
        xp1 = val
        yp2 = yp1
        yp1 = out
    
    # Combine
    for j in range(H):
        idx = row * H + j
        yy1_val = tl.load(yy1 + idx)
        y2_val = tl.load(y2 + idx)
        result = c1 * (yy1_val + y2_val)
        tl.store(imgOut + idx, result)

@triton.jit
def deriche_vertical_kernel(imgOut, yy1, y2,
                           a5, a6, a7, a8, b1, b2, c2,
                           H: tl.constexpr, W: tl.constexpr):
    col = tl.program_id(0)
    
    # Forward pass
    tm1 = 0.0
    ym1 = 0.0
    ym2 = 0.0
    for i in range(W):
        idx = i * H + col
        val = tl.load(imgOut + idx)
        out = a5 * val + a6 * tm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1 + idx, out)
        tm1 = val
        ym2 = ym1
        ym1 = out
    
    # Backward pass
    tp1 = 0.0
    tp2 = 0.0
    yp1 = 0.0
    yp2 = 0.0
    for i in range(W-1, -1, -1):
        idx = i * H + col
        val = tl.load(imgOut + idx)
        out = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
        tl.store(y2 + idx, out)
        tp2 = tp1
        tp1 = val
        yp2 = yp1
        yp1 = out
    
    # Combine
    for i in range(W):
        idx = i * H + col
        yy1_val = tl.load(yy1 + idx)
        y2_val = tl.load(y2 + idx)
        result = c2 * (yy1_val + y2_val)
        tl.store(imgOut + idx, result)

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    # Compute coefficients
    exp_alpha = math.exp(-alpha)
    exp_2alpha = math.exp(-2.0 * alpha)
    
    k = (1.0 - exp_alpha) * (1.0 - exp_alpha) / (1.0 + 2.0 * alpha * exp_alpha - exp_2alpha)
    a1 = a5 = k
    a2 = a6 = k * exp_alpha * (alpha - 1.0)
    a3 = a7 = k * exp_alpha * (alpha + 1.0)
    a4 = a8 = -k * exp_2alpha
    b1 = 2.0 ** (-alpha)
    b2 = -exp_2alpha
    c1 = c2 = 1.0
    
    # Horizontal processing (W rows in parallel)
    deriche_horizontal_kernel[(W,)](
        imgOut, imgIn, yy1, y2,
        a1, a2, a3, a4, b1, b2, c1,
        H, W
    )
    
    # Vertical processing (H columns in parallel)
    deriche_vertical_kernel[(H,)](
        imgOut, yy1, y2,
        a5, a6, a7, a8, b1, b2, c2,
        H, W
    )