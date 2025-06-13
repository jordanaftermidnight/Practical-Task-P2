"""Advanced Matrix Operations with NumPy"""
import numpy as np
from scipy.linalg import solve, lstsq, pinv, norm
from typing import Tuple, Optional, Union, List
import warnings

class MatrixOperations:
    """
    Advanced matrix operations and linear algebra computations.
    """
    
    def __init__(self, precision: str = 'float64'):
        self.precision = np.dtype(precision)
        
    def matrix_decompositions(self, matrix: np.ndarray) -> dict:
        """
        Perform various matrix decompositions.
        """
        results = {}
        
        # Ensure matrix is square for some decompositions
        m, n = matrix.shape
        
        # LU Decomposition
        try:
            from scipy.linalg import lu
            P, L, U = lu(matrix)
            results['lu'] = {'P': P, 'L': L, 'U': U}
            
            # Verify: P @ matrix = L @ U
            results['lu']['verification'] = np.allclose(P @ matrix, L @ U)
        except Exception as e:
            results['lu'] = {'error': str(e)}
        
        # QR Decomposition
        try:
            Q, R = np.linalg.qr(matrix)
            results['qr'] = {'Q': Q, 'R': R}
            
            # Verify: matrix = Q @ R
            results['qr']['verification'] = np.allclose(matrix, Q @ R)
        except Exception as e:
            results['qr'] = {'error': str(e)}
        
        # Singular Value Decomposition (SVD)
        try:
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            results['svd'] = {'U': U, 's': s, 'Vt': Vt}
            
            # Verify: matrix = U @ diag(s) @ Vt
            reconstructed = U @ np.diag(s) @ Vt
            results['svd']['verification'] = np.allclose(matrix, reconstructed)
            
            # Condition number and rank
            results['svd']['condition_number'] = s[0] / s[-1] if s[-1] != 0 else np.inf
            results['svd']['numerical_rank'] = np.sum(s > 1e-12)
        except Exception as e:
            results['svd'] = {'error': str(e)}
        
        # Cholesky Decomposition (for positive definite matrices)
        if m == n:
            try:
                # Make matrix positive definite
                symmetric_matrix = matrix.T @ matrix + np.eye(n) * 1e-6
                L_chol = np.linalg.cholesky(symmetric_matrix)
                results['cholesky'] = {'L': L_chol}
                
                # Verify: symmetric_matrix = L @ L.T
                results['cholesky']['verification'] = np.allclose(symmetric_matrix, L_chol @ L_chol.T)
            except Exception as e:
                results['cholesky'] = {'error': str(e)}
        
        # Eigenvalue Decomposition (for square matrices)
        if m == n:
            try:
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                results['eigen'] = {
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors
                }
                
                # Sort by eigenvalue magnitude
                idx = np.argsort(np.abs(eigenvalues))[::-1]
                results['eigen']['sorted_eigenvalues'] = eigenvalues[idx]
                results['eigen']['sorted_eigenvectors'] = eigenvectors[:, idx]
                
                # Verify eigenvalue equation: A @ v = λ @ v
                verifications = []
                for i in range(min(3, len(eigenvalues))):  # Check first 3
                    v = eigenvectors[:, i]
                    lam = eigenvalues[i]
                    verification = np.allclose(matrix @ v, lam * v)
                    verifications.append(verification)
                results['eigen']['verification'] = verifications
                
            except Exception as e:
                results['eigen'] = {'error': str(e)}
        
        return results
    
    def advanced_linear_systems(self, A: np.ndarray, b: np.ndarray) -> dict:
        """
        Solve linear systems using various methods.
        """
        results = {}
        
        # Direct solution (if square and well-conditioned)
        try:
            if A.shape[0] == A.shape[1]:
                x_direct = solve(A, b)
                results['direct_solution'] = {
                    'x': x_direct,
                    'residual': norm(A @ x_direct - b),
                    'method': 'LU factorization'
                }
        except Exception as e:
            results['direct_solution'] = {'error': str(e)}
        
        # Least squares solution (for overdetermined systems)
        try:
            x_lstsq, residuals, rank, s = lstsq(A, b)
            results['least_squares'] = {
                'x': x_lstsq,
                'residuals': residuals,
                'rank': rank,
                'singular_values': s,
                'condition_number': s[0] / s[-1] if len(s) > 0 and s[-1] != 0 else np.inf
            }
        except Exception as e:
            results['least_squares'] = {'error': str(e)}
        
        # Pseudoinverse solution
        try:
            A_pinv = pinv(A)
            x_pinv = A_pinv @ b
            results['pseudoinverse'] = {
                'x': x_pinv,
                'residual': norm(A @ x_pinv - b),
                'pseudoinverse': A_pinv
            }
        except Exception as e:
            results['pseudoinverse'] = {'error': str(e)}
        
        # Regularized solution (Ridge regression)
        try:
            alpha = 1e-6  # Regularization parameter
            A_reg = A.T @ A + alpha * np.eye(A.shape[1])
            b_reg = A.T @ b
            x_ridge = solve(A_reg, b_reg)
            results['ridge_regression'] = {
                'x': x_ridge,
                'residual': norm(A @ x_ridge - b),
                'regularization_param': alpha
            }
        except Exception as e:
            results['ridge_regression'] = {'error': str(e)}
        
        # Iterative solution using conjugate gradient (for symmetric positive definite)
        try:
            if A.shape[0] == A.shape[1]:
                # Make A symmetric positive definite
                A_spd = A.T @ A + np.eye(A.shape[0]) * 1e-6
                b_spd = A.T @ b
                
                x_cg = self._conjugate_gradient(A_spd, b_spd)
                results['conjugate_gradient'] = {
                    'x': x_cg,
                    'residual': norm(A_spd @ x_cg - b_spd),
                    'original_residual': norm(A @ x_cg - b)
                }
        except Exception as e:
            results['conjugate_gradient'] = {'error': str(e)}
        
        return results
    
    def _conjugate_gradient(self, A: np.ndarray, b: np.ndarray, 
                           x0: Optional[np.ndarray] = None, 
                           maxiter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Conjugate gradient method for solving Ax = b.
        """
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        r = b - A @ x
        p = r.copy()
        rsold = r.T @ r
        
        for i in range(maxiter):
            Ap = A @ p
            alpha = rsold / (p.T @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.T @ r
            
            if np.sqrt(rsnew) < tol:
                break
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def matrix_functions(self, matrix: np.ndarray) -> dict:
        """
        Compute various matrix functions.
        """
        results = {}
        
        # Matrix exponential
        try:
            from scipy.linalg import expm
            mat_exp = expm(matrix)
            results['exponential'] = mat_exp
        except Exception as e:
            results['exponential'] = {'error': str(e)}
        
        # Matrix logarithm
        try:
            from scipy.linalg import logm
            mat_log = logm(matrix)
            results['logarithm'] = mat_log
        except Exception as e:
            results['logarithm'] = {'error': str(e)}
        
        # Matrix square root
        try:
            from scipy.linalg import sqrtm
            mat_sqrt = sqrtm(matrix)
            results['square_root'] = mat_sqrt
        except Exception as e:
            results['square_root'] = {'error': str(e)}
        
        # Matrix power
        try:
            powers = [0.5, 2, -1, -0.5]
            matrix_powers = {}
            for p in powers:
                if p == -1:
                    # Matrix inverse
                    matrix_powers[p] = np.linalg.inv(matrix)
                else:
                    # General matrix power using eigendecomposition
                    eigenvals, eigenvecs = np.linalg.eig(matrix)
                    # Handle complex eigenvalues
                    powered_eigenvals = np.power(eigenvals, p)
                    matrix_powers[p] = eigenvecs @ np.diag(powered_eigenvals) @ np.linalg.inv(eigenvecs)
            
            results['powers'] = matrix_powers
        except Exception as e:
            results['powers'] = {'error': str(e)}
        
        # Polar decomposition
        try:
            U, s, Vt = np.linalg.svd(matrix)
            # Unitary part
            Q = U @ Vt
            # Positive semidefinite part
            P = Vt.T @ np.diag(s) @ Vt
            
            results['polar'] = {
                'unitary': Q,
                'positive_semidefinite': P,
                'verification': np.allclose(matrix, Q @ P)
            }
        except Exception as e:
            results['polar'] = {'error': str(e)}
        
        return results
    
    def advanced_matrix_analysis(self, matrix: np.ndarray) -> dict:
        """
        Perform comprehensive matrix analysis.
        """
        results = {}
        
        # Basic properties
        results['shape'] = matrix.shape
        results['dtype'] = matrix.dtype
        results['memory_usage'] = matrix.nbytes
        
        # Norms
        results['norms'] = {
            'frobenius': np.linalg.norm(matrix, 'fro'),
            'nuclear': np.sum(np.linalg.svd(matrix, compute_uv=False)),
            'spectral': np.linalg.norm(matrix, 2),
            'max': np.linalg.norm(matrix, np.inf),
            'min': np.linalg.norm(matrix, -np.inf)
        }
        
        # Condition numbers
        try:
            results['condition_numbers'] = {
                'l2': np.linalg.cond(matrix, 2),
                'frobenius': np.linalg.cond(matrix, 'fro'),
                'inf': np.linalg.cond(matrix, np.inf)
            }
        except:
            results['condition_numbers'] = {'error': 'Could not compute condition numbers'}
        
        # Matrix properties
        is_square = matrix.shape[0] == matrix.shape[1]
        results['properties'] = {
            'is_square': is_square,
            'is_symmetric': is_square and np.allclose(matrix, matrix.T),
            'is_hermitian': is_square and np.allclose(matrix, matrix.conj().T),
            'is_orthogonal': is_square and np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0])),
            'is_positive_definite': False,
            'is_diagonally_dominant': False
        }
        
        if is_square:
            # Check positive definiteness
            try:
                eigenvals = np.linalg.eigvals(matrix)
                results['properties']['is_positive_definite'] = np.all(eigenvals > 0)
            except:
                pass
            
            # Check diagonal dominance
            diag_elements = np.abs(np.diag(matrix))
            off_diag_sums = np.sum(np.abs(matrix), axis=1) - diag_elements
            results['properties']['is_diagonally_dominant'] = np.all(diag_elements >= off_diag_sums)
        
        # Rank and nullity
        try:
            _, s, _ = np.linalg.svd(matrix)
            numerical_rank = np.sum(s > 1e-12)
            results['rank_analysis'] = {
                'numerical_rank': numerical_rank,
                'theoretical_rank': min(matrix.shape),
                'nullity': matrix.shape[1] - numerical_rank,
                'singular_values': s
            }
        except Exception as e:
            results['rank_analysis'] = {'error': str(e)}
        
        # Sparsity analysis
        zero_elements = np.sum(np.abs(matrix) < 1e-12)
        total_elements = matrix.size
        results['sparsity'] = {
            'zero_elements': zero_elements,
            'nonzero_elements': total_elements - zero_elements,
            'sparsity_ratio': zero_elements / total_elements,
            'density': (total_elements - zero_elements) / total_elements
        }
        
        return results