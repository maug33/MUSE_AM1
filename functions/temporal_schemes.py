import numpy as np

def euler_step(u, t, f, dt, **kwargs):
    """
    Explicit Euler step: U_{n+1} = U_n + dt * F(U_n, t_n)
    Order: 1
    """
    return u + dt * f(u, t, **kwargs)

def rk4_step(u, t, f, dt, **kwargs):
    """
    Runge-Kutta 4th Order step.
    Order: 4
    """
    k1 = dt * f(u, t, **kwargs)
    k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt, **kwargs)
    k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt, **kwargs)
    k4 = dt * f(u + k3, t + dt, **kwargs)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6

def inverse_euler_step(u, t, f, dt, jacobian_f=None, tol=1e-10, max_iter=20, **kwargs):
    """
    Implicit Inverse Euler using Newton-Raphson.
    Order: 1
    Stability: L-stable (good for stiff problems)
    """
    # Initial guess using explicit Euler
    u_guess = u + dt * f(u, t, **kwargs)

    for _ in range(max_iter):
        f_val = f(u_guess, t + dt, **kwargs)
        # Root finding for G(u_new) = u_new - u - dt*F(u_new) = 0
        G = u_guess - u - dt * f_val

        if np.linalg.norm(G) < tol:
            return u_guess

        # Use analytical Jacobian if provided, else numerical
        if jacobian_f:
            J_f = jacobian_f(u_guess, t + dt, **kwargs)
        else:
            J_f = get_numerical_jacobian(f, u_guess, t + dt, **kwargs)

        # Jacobian of G: I - dt * J_f
        J_G = np.eye(len(u)) - dt * J_f

        try:
            delta = np.linalg.solve(J_G, -G)
        except np.linalg.LinAlgError:
            return u_guess # Fallback if singular

        u_guess = u_guess + delta

    return u_guess

def crank_nicolson_step(u, t, f, dt, jacobian_f=None, tol=1e-10, max_iter=20, **kwargs):
    """
    Implicit Crank-Nicolson using Newton-Raphson.
    Order: 2
    Stability: A-stable (conserves energy better than Euler)
    """
    f_current = f(u, t, **kwargs)
    u_guess = u + dt * f_current # Explicit guess

    for _ in range(max_iter):
        f_next = f(u_guess, t + dt, **kwargs)
        # Root finding for G = u_new - u - 0.5*dt*(F_n + F_new) = 0
        G = u_guess - u - 0.5 * dt * (f_current + f_next)

        if np.linalg.norm(G) < tol:
            return u_guess

        # Determine Jacobian
        if jacobian_f:
            J_f_next = jacobian_f(u_guess, t + dt, **kwargs)
        else:
            J_f_next = get_numerical_jacobian(f, u_guess, t + dt, **kwargs)

        # Jacobian of G: I - 0.5 * dt * J_f_next
        J_G = np.eye(len(u)) - 0.5 * dt * J_f_next

        try:
            delta = np.linalg.solve(J_G, -G)
        except np.linalg.LinAlgError:
            return u_guess

        u_guess = u_guess + delta

    return u_guess

def integrate(temporal_scheme, u0, t0, tf, f, dt, **kwargs):
    """
    General integration driver for Cauchy problems.

    Args:
        temporal_scheme: Function implementing the stepping (e.g., rk4_step)
        u0: Initial state vector
        t0: Start time
        tf: Final time
        f: Differential equation F(u, t)
        dt: Time step
        **kwargs: Passed to the stepper (e.g., jacobian_f)
    """
    times = np.arange(t0, tf + dt, dt)
    # Ensure the last step hits exactly tf if needed, or strictly adhere to dt
    # Here we strictly adhere to dt steps defined by arange

    trajectory = np.zeros((len(times), len(u0)))
    trajectory[0] = u0

    u = u0
    for i in range(len(times) - 1):
        u = temporal_scheme(u, times[i], f, dt, **kwargs)
        trajectory[i+1] = u

    return trajectory, times

def get_numerical_jacobian(f, u, t, epsilon=1e-8, **kwargs):
    """
    Computes Jacobian J = dF/dU numerically via central finite differences.
    This works for ANY differential equation function 'f'.
    """
    n = len(u)
    J = np.zeros((n, n))
    
    # Iterate over each variable to calculate partial derivatives
    for i in range(n):
        u_plus = u.copy()
        u_minus = u.copy()
        
        # Perturb the i-th component up and down
        u_plus[i] += epsilon
        u_minus[i] -= epsilon
        
        # Evaluate F at the perturbed points
        f_plus = f(u_plus, t, **kwargs)
        f_minus = f(u_minus, t, **kwargs)
        
        # Central Difference: df/du ≈ (f(u+e) - f(u-e)) / 2e
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
    return J