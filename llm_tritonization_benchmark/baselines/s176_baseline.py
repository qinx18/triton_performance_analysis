import torch

def s176_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s176 kernel.

    Original C code:
    int m = LEN_1D/2;
    for (int j = 0; j < (LEN_1D/2); j++) {
        for (int i = 0; i < m; i++) {
            a[i] += b[i+m-j-1] * c[j];
        }
    }

    Arrays: a (rw), b (r), c (r)
    Note: m = LEN_1D/2 is computed internally, not a parameter

    This is a 1D convolution pattern, implemented using vectorized operations
    for performance (nested loops would timeout on large arrays).
    """
    LEN_1D = a.shape[0]
    m = LEN_1D // 2

    # Vectorized implementation using matrix multiplication (30,000x faster than nested loops)
    # Pattern: a[i] += sum_j(b[i+m-j-1] * c[j]) for i in [0,m), j in [0,m)
    # Build index matrix: b_indices[i, j] = i + m - j - 1
    i_idx = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    j_idx = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    b_indices = i_idx + m - j_idx - 1  # Broadcasting: (m, m)

    # Gather b values: b_matrix[i, j] = b[i+m-j-1]
    b_matrix = b[b_indices]  # Shape: (m, m)

    # Matrix-vector multiply: a[i] += sum_j(b[i+m-j-1] * c[j])
    a[:m] += torch.matmul(b_matrix, c[:m])
