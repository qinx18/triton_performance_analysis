#!/usr/bin/env python3
"""
Baseline: Sparse Matrix-Vector Multiplication (SpMV) using PyTorch

SpMV is a fundamental operation: y = A @ x
where A is a sparse matrix (in CSR format) and x is a dense vector.

PyTorch sparse tensors use COO format by default but can be converted to CSR.
"""

import torch

def create_random_sparse_matrix(rows, cols, sparsity=0.95, device='cuda', dtype=torch.float32):
    """
    Create a random sparse matrix in CSR format

    Args:
        rows: Number of rows
        cols: Number of columns
        sparsity: Fraction of elements that are zero (0.95 = 95% sparse)
        device: Device to create tensor on
        dtype: Data type

    Returns:
        Sparse tensor in CSR format
    """
    # Calculate number of non-zero elements
    nnz = int(rows * cols * (1 - sparsity))

    # Create random indices for non-zero elements
    indices = torch.randint(0, rows * cols, (nnz,), device='cpu')
    indices = torch.unique(indices)  # Remove duplicates

    # Convert flat indices to 2D coordinates
    row_indices = indices // cols
    col_indices = indices % cols

    # Create random values
    values = torch.randn(len(indices), dtype=dtype, device='cpu')

    # Create COO sparse tensor
    indices_2d = torch.stack([row_indices, col_indices])
    sparse_coo = torch.sparse_coo_tensor(
        indices_2d, values, (rows, cols), dtype=dtype, device='cpu'
    )

    # Convert to CSR format (more efficient for SpMV)
    sparse_csr = sparse_coo.to_sparse_csr().to(device)

    return sparse_csr


def spmv_baseline(sparse_matrix, dense_vector):
    """
    Sparse Matrix-Vector multiplication using PyTorch

    Args:
        sparse_matrix: Sparse matrix in CSR format (M x N)
        dense_vector: Dense vector (N,)

    Returns:
        Dense result vector (M,)
    """
    # PyTorch sparse @ operation handles CSR efficiently
    return torch.sparse.mm(sparse_matrix, dense_vector.unsqueeze(1)).squeeze(1)


def spmv_baseline_coo(sparse_matrix_coo, dense_vector):
    """
    Alternative: SpMV using COO format

    Args:
        sparse_matrix_coo: Sparse matrix in COO format (M x N)
        dense_vector: Dense vector (N,)

    Returns:
        Dense result vector (M,)
    """
    return torch.sparse.mm(sparse_matrix_coo, dense_vector.unsqueeze(1)).squeeze(1)


def get_csr_arrays(sparse_csr):
    """
    Extract CSR format arrays from PyTorch sparse tensor

    CSR format consists of:
    - values: Non-zero values
    - col_indices: Column index for each non-zero value
    - row_ptr: Pointer to start of each row in values array

    Returns:
        tuple: (values, col_indices, row_ptr, num_rows, num_cols)
    """
    values = sparse_csr.values()
    col_indices = sparse_csr.col_indices()
    row_ptr = sparse_csr.crow_indices()  # crow = compressed row
    num_rows, num_cols = sparse_csr.shape

    return values, col_indices, row_ptr, num_rows, num_cols


if __name__ == "__main__":
    print("="*60)
    print("Testing Sparse Matrix-Vector Multiplication Baseline")
    print("="*60)

    # Test configuration
    M, N = 4096, 4096
    sparsity = 0.95  # 95% sparse

    print(f"\nMatrix size: {M} x {N}")
    print(f"Sparsity: {sparsity*100:.1f}%")

    # Create sparse matrix
    A_csr = create_random_sparse_matrix(M, N, sparsity=sparsity)

    # Get CSR arrays
    values, col_indices, row_ptr, num_rows, num_cols = get_csr_arrays(A_csr)

    print(f"Non-zero elements: {len(values):,}")
    print(f"Storage efficiency: {len(values)/(M*N)*100:.2f}% of dense")

    # Create dense vector
    x = torch.randn(N, device='cuda', dtype=torch.float32)

    # Perform SpMV
    y = spmv_baseline(A_csr, x)

    print(f"\nInput vector shape: {x.shape}")
    print(f"Output vector shape: {y.shape}")
    print(f"Output mean: {y.mean():.6f}")
    print(f"Output std: {y.std():.6f}")

    # Verify correctness by converting to dense
    A_dense = A_csr.to_dense()
    y_dense = A_dense @ x
    max_diff = (y - y_dense).abs().max().item()

    print(f"\nVerification against dense multiplication:")
    print(f"Max difference: {max_diff:.10f}")
    print(f"Correct: {max_diff < 1e-5}")
