"""Optional numpy reference path for small geometry sanity checks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def skew_projection(matrix: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, matrix - np.swapaxes(matrix, -1, -2))


def matrix_exp_reference(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    values, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    result = vectors @ np.diag(np.exp(values)) @ inverse
    return np.real_if_close(result).astype(np.float64)


def loop_holonomy_reference(connection_matrices: np.ndarray, loop_edges: list[int]) -> np.ndarray:
    if not loop_edges:
        return np.eye(connection_matrices.shape[-1], dtype=np.float64)
    transport = np.eye(connection_matrices.shape[-1], dtype=np.float64)
    for edge in loop_edges:
        transport = matrix_exp_reference(connection_matrices[edge]) @ transport
    return transport


def observables_reference(holonomy: np.ndarray) -> dict[str, Any]:
    fiber_dim = holonomy.shape[-1]
    identity = np.eye(fiber_dim, dtype=np.float64)
    delta = holonomy - identity
    eigenvalues = np.linalg.eigvals(holonomy)
    return {
        "holonomy_norm": float(np.linalg.norm(delta, ord="fro")),
        "trace_normalized": float(np.trace(holonomy) / fiber_dim),
        "spectrum_abs_mean": float(np.abs(eigenvalues).mean()),
        "spectrum_abs_max": float(np.abs(eigenvalues).max()),
    }
