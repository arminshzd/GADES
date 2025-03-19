import jax.numpy as jnp
from jax import jit

@jit
def _get_spectral_abs(A):
    """
    Given a matrix `A` calculates the __spectral__ absolute matrix `|A|` where the the eigenvalues of `A` and `|A|` have the same magnitude, but they are all positive for `|A|`.

    Args:
        A (jax.ndarray): NxN matrix

    Returns:
        jax.ndarray: NxN matrix |A| with all positive eigenvalues.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(A)
    
    # Take the absolute value of the eigenvalues
    abs_eigenvalues = jnp.abs(eigenvalues)
    
    # Reconstruct the matrix using the eigenvectors and absolute eigenvalues
    abs_A = eigenvectors @ jnp.diag(abs_eigenvalues) @ jnp.linalg.inv(eigenvectors)
    
    isreal = jnp.all(jnp.where(abs_A.imag == 0, True, False))
    return jnp.where(isreal, abs_A, jnp.zeros_like(abs_A))

@jit
def get_bofill_H(pos_n, pos_o, grad_n, grad_o, H):
    """
    Bofill's Hessian estimation algorithm based on https://doi.org/10.1002/qua.10709 (TS-BFGS method)

    Args:
        pos_n (jax.ndarray): 3Nx1 vector of positions at t+dt
        pos_o (jax.ndarray): 3Nx1 vector of positions at t
        grad_n (jax.ndarray): 3Nx1 vector of gradiants at t+dt
        grad_o (jax.ndarray): 3Nx1 vector of gradiants at t
        H (jax.ndarray): 3Nx3N Hessian matrix at t

    Returns:
        jax.ndarray: 3Nx3N estimated Hessian matrix at t+dt
    """
    # calculating diff vectors
    d_vec = pos_n - pos_o
    y_vec = grad_n - grad_o
    # enforcing flattened column vectors
    d_vec = d_vec.reshape(-1, 1) # 3N x 1
    y_vec = y_vec.reshape(-1, 1) # 3N x 1
    # calculating update mat E
    abs_H = jnp.real(_get_spectral_abs(H)) # 3N x 3N
    # E = A + B - C
    # A = A1@A2.T/A3
    A1 = y_vec - (H@d_vec) # vec
    # A2 = A21*y_vec + A22@abs_H@d_vec
    A21 = y_vec.T@d_vec
    A22 = d_vec.T@abs_H@d_vec
    A2 = A21*y_vec + A22*(abs_H@d_vec) # vec
    # A3 = A21^2 + A22^2
    A3 = (A21**2) + (A22**2) # scalar
    A = (1/A3) * (A1@A2.T) # mat
    # B = A2@A1.T/A3
    B = (1/A3) * (A2@A1.T) # mat
    # C = (C1/A3^2)*(A2@A2.T)
    C1 = (y_vec.T@d_vec) - (d_vec.T@H@d_vec)
    C = (C1/A3**2)*(A2@A2.T)
    E = A + B - C
    return H + E