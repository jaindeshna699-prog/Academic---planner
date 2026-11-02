"""
Module: power_method.py
Implements the Power Method from scratch to compute the dominant eigenvalue
and principal eigenvector of a subject-priority matrix.
Used to rank subjects in order of study importance based on credits,
weakness ratings, syllabus backlog, and exam urgency.
Verified against numpy.linalg.eig.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np


def power_method(
    A: Union[List[List[float]], np.ndarray],
    v0: Optional[Union[List[float], np.ndarray]] = None,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Computes the dominant eigenvalue and eigenvector of matrix A using the Power Method from scratch.

    Method:
        Iterative power iteration with Rayleigh Quotient estimation:
        y_k = A * v_{k-1}
        v_k = y_k / ||y_k||_2
        \lambda_k = (v_k^T * A * v_k) / (v_k^T * v_k)
    Complexity:
        Time Complexity: O(k * n^2) where k is iteration count and n is matrix size.
        Space Complexity: O(n) for state vectors.

    Parameters:
        A: Square real matrix (n x n).
        v0: Initial non-zero guess vector. If None, initialized to uniform ones.
        tol: Convergence tolerance for eigenvalue change |\lambda_k - \lambda_{k-1}|.
        max_iter: Maximum permitted iterations.

    Returns:
        dict containing:
            - eigenvalue: Dominant eigenvalue \lambda_1.
            - eigenvector: Normalized dominant eigenvector v_1 (sum=1 or norm=1).
            - iterations: Number of iterations performed.
            - converged: Boolean status.
            - iteration_history: Table of iteration steps.
    """
    A_mat = np.array(A, dtype=float, copy=True)
    n = A_mat.shape[0]

    if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Matrix A must be 2D square matrix, received shape: {A_mat.shape}.")

    if v0 is None:
        v = np.ones(n, dtype=float) / np.sqrt(n)
    else:
        v = np.array(v0, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            raise ValueError("Initial guess vector v0 cannot be the zero vector.")
        v = v / norm_v

    lambda_old = 0.0
    history: List[Dict[str, Any]] = []

    for k in range(1, max_iter + 1):
        # Matrix-vector multiplication y = A * v
        y = np.dot(A_mat, v)
        norm_y = np.linalg.norm(y)

        if norm_y < 1e-14:
            raise ValueError("Iteration vector collapsed to zero: matrix may be nilpotent or singular.")

        # Normalize
        v_next = y / norm_y

        # Rayleigh quotient
        lambda_k = float(np.dot(v_next, np.dot(A_mat, v_next)))
        err = abs(lambda_k - lambda_old)

        history.append({
            "iteration": k,
            "eigenvalue_estimate": float(lambda_k),
            "step_error": float(err),
            "eigenvector_norm": float(norm_y),
            "v_components": [float(x) for x in v_next]
        })

        if err < tol and k > 1:
            # Positive normalized eigenvector (sum to 1.0 for priority weights)
            v_normed = v_next if np.sum(v_next) >= 0 else -v_next
            v_positive = np.abs(v_normed)
            sum_v = np.sum(v_positive)
            weights = (v_positive / sum_v) if sum_v > 0 else v_positive

            return {
                "eigenvalue": float(lambda_k),
                "eigenvector": [float(val) for val in weights],
                "eigenvector_unit": [float(val) for val in v_next],
                "iterations": k,
                "converged": True,
                "iteration_history": history
            }

        v = v_next
        lambda_old = lambda_k

    v_positive = np.abs(v)
    sum_v = np.sum(v_positive)
    weights = (v_positive / sum_v) if sum_v > 0 else v_positive

    return {
        "eigenvalue": float(lambda_old),
        "eigenvector": [float(val) for val in weights],
        "eigenvector_unit": [float(val) for val in v],
        "iterations": max_iter,
        "converged": False,
        "iteration_history": history
    }


def build_subject_priority_matrix(
    subject_names: List[str],
    credits_list: List[float],
    strength_ratings: List[int],  # 1 (weakest) to 5 (strongest)
    days_to_exam: List[int],
    syllabus_pending_pct: List[float],
    dependency_graph: Optional[Dict[int, List[int]]] = None
) -> Dict[str, Any]:
    """
    Constructs a positive pairwise comparison priority matrix based on:
        - Credits (c_i)
        - Difficulty/Weakness: d_i = (6 - strength_i)
        - Exam Urgency: u_i = 1.0 / sqrt(max(1, days_to_exam))
        - Syllabus Pending: p_i = max(0.1, syllabus_pending_pct / 100)
    Applies Power Method to obtain the dominant eigenvector representing study priorities.
    """
    n = len(subject_names)
    if n == 0:
        raise ValueError("Must provide at least one subject.")
    if n == 1:
        return {
            "ranking": [(subject_names[0], 1.0)],
            "matrix": [[1.0]],
            "eigenvalue": 1.0,
            "eigenvector": [1.0]
        }

    # Raw composite score for each subject
    composite_scores = []
    for i in range(n):
        c = float(credits_list[i])
        diff = float(6 - max(1, min(5, strength_ratings[i])))
        urgency = 1.0 / np.sqrt(max(1.0, float(days_to_exam[i])))
        pending = max(0.1, float(syllabus_pending_pct[i]) / 100.0)
        score = c * diff * pending * urgency
        composite_scores.append(max(1e-4, score))

    # Build pairwise matrix A[i, j] = score[i] / score[j] (Perron-Frobenius positive matrix)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            A[i, j] = composite_scores[i] / composite_scores[j]

    # Incorporate prerequisite/dependency coupling if provided
    if dependency_graph:
        for parent, prereqs in dependency_graph.items():
            for child in prereqs:
                if 0 <= parent < n and 0 <= child < n:
                    A[parent, child] *= 1.25  # Prerequisite gets boost
                    A[child, parent] = 1.0 / A[parent, child]

    res = power_method(A)
    weights = res["eigenvector"]

    # Rank subjects
    ranked = sorted(
        [(name, float(w)) for name, w in zip(subject_names, weights)],
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "ranking": ranked,
        "priority_matrix": A.tolist(),
        "eigenvalue": res["eigenvalue"],
        "eigenvector": weights,
        "method_details": res
    }


def benchmark_power_method(A: List[List[float]]) -> Dict[str, Any]:
    """
    Benchmarks from-scratch Power Method against numpy.linalg.eig.
    """
    scratch_res = power_method(A)

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Find dominant eigenvalue (largest absolute value)
        dominant_idx = int(np.argmax(np.abs(eigenvalues)))
        np_dominant_eval = float(np.real(eigenvalues[dominant_idx]))

        # Dominant eigenvector
        np_dom_evec = np.real(eigenvectors[:, dominant_idx])
        np_evec_pos = np.abs(np_dom_evec)
        np_weights = [float(x) for x in (np_evec_pos / np.sum(np_evec_pos))]

        np_status = "Success"
        eval_diff = abs(scratch_res["eigenvalue"] - np_dominant_eval)
        evec_diff = [abs(s - n_w) for s, n_w in zip(scratch_res["eigenvector"], np_weights)]
        max_evec_diff = max(evec_diff)
    except Exception as e:
        np_dominant_eval = None
        np_weights = None
        np_status = f"NumPy Error: {str(e)}"
        eval_diff = None
        max_evec_diff = None

    return {
        "scratch_eigenvalue": scratch_res["eigenvalue"],
        "numpy_eigenvalue": np_dominant_eval,
        "eigenvalue_difference": eval_diff,
        "scratch_weights": scratch_res["eigenvector"],
        "numpy_weights": np_weights,
        "max_weight_difference": max_evec_diff,
        "numpy_status": np_status,
        "iterations": scratch_res["iterations"]
    }
