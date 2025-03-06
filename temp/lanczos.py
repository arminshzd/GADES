from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, grad, jvp, hessian
from jax.scipy.sparse.linalg import cg
import time

@partial(jit, static_argnums=(0,))
def lanczos_HVP(loss_fn, params, num_iter=100, key=jax.random.PRNGKey(0)):
    grad_fn = grad(loss_fn)  # Precompute gradient function
    
    def hvp(v):
        _, hvp_val = jvp(grad_fn, (params,), (v,))
        return hvp_val

    n = params.shape[0]
    ##### TEST
    num_iter = 10
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
def lanczos_HVP_SaI(loss_fn, params, sig, num_iter=100, key=jax.random.PRNGKey(0)):
    grad_fn = grad(loss_fn)  # Precompute gradient function
    
    def hvp(v):
        _, hvp_val = jvp(grad_fn, (params,), (v,))
        return hvp_val
    
    def SaI_on_w(v, max_iter=100, tol=1e-6):
        """
        Compute Bv = (A - cI)^(-1)v using iterative methods.

        Args:
            v (jax.numpy.ndarray): The input vector v.
            max_iter (int): Maximum number of iterations for CG solver. Default is 100.
            tol (float): Tolerance for convergence. Default is 1e-6.

        Returns:
            jax.numpy.ndarray: The result Bv = (A - cI)^(-1)v.
        """

        # Define the linear operator for (A - cI)w
        def matvec(x):
            return hvp(x) - (sig * x)

        # Use Conjugate Gradient to solve the system: (A - cI)x = v
        solution, _ = cg(matvec, v, tol=tol, maxiter=max_iter)
        
        return solution

    n = params.shape[0]
    ##### TEST
    num_iter = 10
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
def lanczos(A, num_iter=100, key=jax.random.PRNGKey(0)):

    n = A.shape[0]
    ##### TEST
    num_iter = 10
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
def lanczos_SaI(A, sig, num_iter=100, key=jax.random.PRNGKey(0)):

    n = A.shape[0]
    ##### TEST
    num_iter = 10
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

if __name__ == "__main__":
    N = 5
    def quadratic_loss(params):
        return 0.5 * jnp.sum(params @ jnp.diag(jnp.arange(-3, N-3)) @ params)

    
    params = jnp.ones(N)

    #print(grad(quadratic_loss)(params))
    #print(hessian(quadratic_loss)(params))

    ts = time.time()
    H_fun = hessian(quadratic_loss)
    H = H_fun(params)
    w, v = jnp.linalg.eigh(H)
    print("actual values:", w)
    print("actual values:", v)
    tf = time.time()
    print("full H time:", tf-ts, "s")

    ts = time.time()
    eigenvalues, eigenvecs = lanczos_HVP(quadratic_loss, params, num_iter=10)
    print("HVP:", eigenvalues)  # Should approach [5.0, 5.0]
    print("HVP:", eigenvecs[:, -1])  # Should approach [5.0, 5.0]
    tf = time.time()
    print("HVP time:", tf-ts, "s")

    ts = time.time()
    eigenvalues, eigenvecs = lanczos_HVP_SaI(quadratic_loss, params, 2.0001, num_iter=10)
    print("HVP SaI:", eigenvalues[0])  # Should approach [5.0, 5.0]
    print("HVP SaI:", eigenvecs[:, 0])  # Should approach [5.0, 5.0]
    tf = time.time()
    print("HVP time:", tf-ts, "s")

    ts = time.time()
    H_fun = hessian(quadratic_loss)
    H = H_fun(params)
    w, v = lanczos(H, 10)
    print("lanczos:", w[-1])
    print("lanczos:", v[:, -1])
    tf = time.time()
    print("lanczos time:", tf-ts, "s")

    ts = time.time()
    H_fun = hessian(quadratic_loss)
    H = H_fun(params)
    w, v = lanczos_SaI(H, 2.0001, 10)
    print("lanczos SaI:", w[0])
    print("lanczos SaI:", v[:, 0])
    tf = time.time()
    print("lanczos time:", tf-ts, "s")