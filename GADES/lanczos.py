from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, grad, jvp
from jax.scipy.sparse.linalg import cg

options = {
    "N_ITER": 10,
    "CG_N_ITER": 100,
    "CG_TOL": 1e-6,
}

@partial(jit, static_argnums=(0,))
def lanczos_HVP(loss_fn, params, key=jax.random.PRNGKey(0)):
    """
    Lanczos method for calculating the dominant eigenvalues and eigenvectors of the hessian of `loss_fn` without explicitly forming it. It calculates the Krylov subpace matrix-vector products through `jax.jvp` instead of explicitly calling `jax.hessian` and diagonalizing it.
    Args:
        loss_fn (callable): AD compatible function R^N -> R
        params (jax.ndarray): (N, ) vector of input params to `loss_fn` 
        key (jax.ndarray): Key for creating the initial random vector. Defaults to jax.random.PRNGKey(0).

    Returns:
        tuple: (eigenvalues, eigenvectors) of `jax.hessian(loss_fn)(params)`
    """

    grad_fn = grad(loss_fn)  # Precompute gradient function
    
    def hvp(v):
        _, hvp_val = jvp(grad_fn, (params,), (v,))
        return hvp_val

    n = params.shape[0]
    num_iter = options["N_ITER"]
    V = jnp.zeros((n, num_iter))
    T = jnp.zeros((num_iter, num_iter))

    # Initialize random unit vector
    v = jax.random.normal(key, (n,))
    v /= jnp.sqrt(jnp.sum(v**2))  
    w = hvp(v)
    alpha = jnp.vdot(v, w)
    w -= alpha * v

    V = V.at[:, 0].set(v)
    T = T.at[0, 0].set(alpha)

    def lanczos_step(j, state):
        V, T, v, w = state

        beta = jnp.sqrt(jnp.sum(w**2))  # More efficient norm calculation
        v_next = w / beta
        w = hvp(v_next)
        w -= beta * v
        alpha = jnp.vdot(v_next, w)
        w -= alpha * v_next

        V = V.at[:, j].set(v_next)
        T = T.at[j, j].set(alpha)
        T = T.at[j, j - 1].set(beta)
        T = T.at[j - 1, j].set(beta)

        return V, T, v_next, w

    V, T, v, w = jax.lax.fori_loop(1, num_iter, lanczos_step, (V, T, v, w))

    # Compute eigenvalues and eigenvectors of T
    eigvals, eigvecs_T = jnp.linalg.eigh(T)

    # Transform eigenvectors of T back to Hessian space
    eigvecs_H = V @ eigvecs_T
    eigvecs_H /= jnp.sqrt(jnp.sum(eigvecs_H**2, axis=0))  # Normalize

    return eigvals, eigvecs_H

@partial(jit, static_argnums=(0,))
def lanczos_HVP_SaI(loss_fn, params, sig, key=jax.random.PRNGKey(0)):
    """
    Lanczos method for calculating the dominant eigenvalues and eigenvectors of the shifted and inverted hessian of `loss_fn` without explicitly forming it. It calculates the Krylov subpace matrix-vector products through `jax.jvp` instead of explicitly calling `jax.hessian` and diagonalizing it. Calculates the eigenvalues and eigenvectors of the shifted and inverted matrix B = (H - sig*I)^(-1) to find the eigenvalues of H that are in the vicinity of `sig`.
    Args:
        loss_fn (callable): AD compatible function R^N -> R
        params (jax.ndarray): (N, ) vector of input params to `loss_fn` 
        sig (float): Shift parameter
        key (jax.ndarray): Key for creating the initial random vector. Defaults to jax.random.PRNGKey(0).

    Returns:
        tuple: (eigenvalues, eigenvectors) of `jax.hessian(loss_fn)(params)`
    """

    grad_fn = grad(loss_fn)  # Precompute gradient function
    
    def hvp(v):
        _, hvp_val = jvp(grad_fn, (params,), (v,))
        return hvp_val
    
    def SaI_on_w(v):
        """
        Compute Bv = (A - cI)^(-1)v using CG.

        Args:
            v (jax.numpy.ndarray): The input vector v.
        Returns:
            jax.numpy.ndarray: The result Bv = (A - cI)^(-1)v.
        """

        # Define the linear operator for (A - cI)x
        def matvec(x):
            return hvp(x) - (sig * x)

        # Use Conjugate Gradient to solve the system: (A - cI)x = v
        solution, _ = cg(matvec, v, tol=options["CG_TOL"], maxiter=options["CG_N_ITER"])
        
        return solution

    n = params.shape[0]
    num_iter = options["N_ITER"]
    V = jnp.zeros((n, num_iter))
    T = jnp.zeros((num_iter, num_iter))

    # Initialize random unit vector
    v = jax.random.normal(key, (n,))
    v /= jnp.sqrt(jnp.sum(v**2))  
    w = SaI_on_w(v)
    alpha = jnp.vdot(v, w)
    w -= alpha * v

    V = V.at[:, 0].set(v)
    T = T.at[0, 0].set(alpha)

    def lanczos_step(j, state):
        V, T, v, w = state

        beta = jnp.sqrt(jnp.sum(w**2))  # More efficient norm calculation
        v_next = w / beta
        w = SaI_on_w(v_next)
        w -= beta * v
        alpha = jnp.vdot(v_next, w)
        w -= alpha * v_next

        V = V.at[:, j].set(v_next)
        T = T.at[j, j].set(alpha)
        T = T.at[j, j - 1].set(beta)
        T = T.at[j - 1, j].set(beta)

        return V, T, v_next, w

    V, T, v, w = jax.lax.fori_loop(1, num_iter, lanczos_step, (V, T, v, w))

    # Compute eigenvalues and eigenvectors of T
    eigvals, eigvecs_T = jnp.linalg.eigh(T)
    eigvals = 1/eigvals + sig


    # Transform eigenvectors of T back to Hessian space
    eigvecs_H = V @ eigvecs_T
    eigvecs_H /= jnp.sqrt(jnp.sum(eigvecs_H**2, axis=0))  # Normalize

    return eigvals, eigvecs_H

@jit
def lanczos(A, key=jax.random.PRNGKey(0)):
    """
    Lanczos method for calculating the dominant eigenvalues and eigenvectors of the matrix `A`.
    Args:
        A (jax.ndarray): (N, N) matrix to calculate eigenvalues and vectors
        key (jax.ndarray): Key for creating the initial random vector. Defaults to jax.random.PRNGKey(0).

    Returns:
        tuple: (eigenvalues, eigenvectors) of `A`
    """

    n = A.shape[0]
    num_iter = options["N_ITER"]
    V = jnp.zeros((n, num_iter))
    T = jnp.zeros((num_iter, num_iter))

    # Initialize random unit vector
    v = jax.random.normal(key, (n,))
    v /= jnp.sqrt(jnp.sum(v**2))  
    w = A @ v
    alpha = jnp.vdot(v, w)
    w -= alpha * v

    V = V.at[:, 0].set(v)
    T = T.at[0, 0].set(alpha)

    def lanczos_step(j, state):
        V, T, v, w = state

        beta = jnp.sqrt(jnp.sum(w**2))  # More efficient norm calculation
        v_next = w / beta
        w = A @ v_next - beta * v
        alpha = jnp.vdot(v_next, w)
        w -= alpha * v_next

        V = V.at[:, j].set(v_next)
        T = T.at[j, j].set(alpha)
        T = T.at[j, j - 1].set(beta)
        T = T.at[j - 1, j].set(beta)

        return V, T, v_next, w

    V, T, v, w = jax.lax.fori_loop(1, num_iter, lanczos_step, (V, T, v, w))

    # Compute eigenvalues and eigenvectors of T
    eigvals, eigvecs_T = jnp.linalg.eigh(T)

    # Transform eigenvectors of T back to Hessian space
    eigvecs_H = V @ eigvecs_T
    eigvecs_H /= jnp.sqrt(jnp.sum(eigvecs_H**2, axis=0))  # Normalize

    return eigvals, eigvecs_H

@jit
def lanczos_SaI(A, sig, key=jax.random.PRNGKey(0)):
    """
    Lanczos method for calculating the dominant eigenvalues and eigenvectors of the shifted and inverted matrix B = (A-sig*I)^(-1). Find the eigenvalues and vectors of `A` closest to `sig`.
    Args:
        A (jax.ndarray): (N, N) matrix to calculate eigenvalues and vectors
        sig (float): Shift parameter
        key (jax.ndarray): Key for creating the initial random vector. Defaults to jax.random.PRNGKey(0).

    Returns:
        tuple: (eigenvalues, eigenvectors) of `A`
    """

    n = A.shape[0]
    num_iter = options["N_ITER"]
    V = jnp.zeros((n, num_iter))
    T = jnp.zeros((num_iter, num_iter))

    A = jnp.linalg.inv(A-jnp.eye(n)*sig)

    # Initialize random unit vector
    v = jax.random.normal(key, (n,))
    v /= jnp.sqrt(jnp.sum(v**2))  
    w = A @ v
    alpha = jnp.vdot(v, w)
    w -= alpha * v

    V = V.at[:, 0].set(v)
    T = T.at[0, 0].set(alpha)

    def lanczos_step(j, state):
        V, T, v, w = state

        beta = jnp.sqrt(jnp.sum(w**2))  # More efficient norm calculation
        v_next = w / beta
        w = A @ v_next - beta * v
        alpha = jnp.vdot(v_next, w)
        w -= alpha * v_next

        V = V.at[:, j].set(v_next)
        T = T.at[j, j].set(alpha)
        T = T.at[j, j - 1].set(beta)
        T = T.at[j - 1, j].set(beta)

        return V, T, v_next, w

    V, T, v, w = jax.lax.fori_loop(1, num_iter, lanczos_step, (V, T, v, w))

    # Compute eigenvalues and eigenvectors of T
    eigvals, eigvecs_T = jnp.linalg.eigh(T)
    eigvals = 1/eigvals + sig

    # Transform eigenvectors of T back to Hessian space
    eigvecs_H = V @ eigvecs_T
    eigvecs_H /= jnp.sqrt(jnp.sum(eigvecs_H**2, axis=0))  # Normalize

    return eigvals, eigvecs_H