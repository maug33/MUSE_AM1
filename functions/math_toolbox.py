import numpy as np

# --- 1. Linear Algebra Solver (Gauss Elimination) ---
def Gauss(A, b):
    """
    Solves Ax = b using Gaussian Elimination with partial pivoting.
    """
    n = len(b)
    # Copy to avoid modifying originals
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    x = np.zeros(n)

    # Forward Elimination
    for k in range(n-1):
        # Partial Pivoting: Find row with max value in column k
        max_index = np.argmax(np.abs(A[k:n, k])) + k
        
        if A[max_index, k] == 0:
            raise ValueError("Matrix is singular!")
            
        # Swap rows in A and b
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
            
        # Eliminate entries below pivot
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back Substitution
    for i in range(n-1, -1, -1):
        sum_ax = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i, i]
        
    return x

# --- 2. Numerical Differentiation ---
def get_jacobian(f, x, epsilon=1e-8, **kwargs):
    """
    Computes the Jacobian Matrix J_ij = df_i/dx_j numerically.
    Accepts **kwargs to pass parameters (like mu) to f.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # We pass **kwargs to the function call here
    f0 = np.array(f(x, **kwargs))
    m = len(f0)
    
    J = np.zeros((m, n))
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        # ...and here!
        f_plus = np.array(f(x_plus, **kwargs))
        f_minus = np.array(f(x_minus, **kwargs))
        
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
    return J

# --- 3. Generic Newton Solver ---
def Newton(F, x0, tol=1e-9, max_iter=50, **kwargs):
    """
    Finds root x such that F(x) = 0.
    Accepts **kwargs to pass parameters (like mu) to F.
    """
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        # 1. Calculate Residual (b = -F(x))
        # Pass kwargs to the function
        fx = np.array(F(x, **kwargs))
        b = -fx
        
        # 2. Check Convergence
        if np.linalg.norm(fx) < tol:
            return x
            
        # 3. Calculate Jacobian
        # Pass kwargs to the jacobian finder (which passes them to F)
        J = get_jacobian(F, x, **kwargs)
        
        # 4. Solve Linear System J * dx = b
        delta_x = Gauss(J, b)
        
        # 5. Update
        x += delta_x
        
    print(f"Warning: Newton did not converge for guess {x0}")
    return x