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


# TO BE REMOVED ONCE OTHER MILESTONES ARE ADJUSTED FOR NEW ORGANIZATION SCHEME
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

# ==========================================
# 5. EMBEDDED RUNGE-KUTTA (ADAPTIVE)
# ==========================================

def get_tableau(name):

    if name == "Bogacki-Shampine":
        #BS3(2) - Order 3 with embedded Order 2
        order = 3
        
        # Nodes (Time steps)
        c = np.array([0, 1/2, 3/4, 1])
        
        # The Runge-Kutta Matrix (A)
        a = np.zeros((4, 4))
        a[1, 0] = 1/2
        a[2, 1] = 3/4
        a[3, 0] = 2/9
        a[3, 1] = 1/3
        a[3, 2] = 4/9
        
        # Weights for the High-Order (3rd) solution
        b = np.array([2/9, 1/3, 4/9, 0])
        
        # Weights for the Low-Order (2nd) solution (for error check)
        bs = np.array([7/24, 1/4, 1/3, 1/8])
    elif name == "Dormand-Prince":
        #DP5(4)- Order 5 with embedded Order 4 (Standard RK45)
        order = 5
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

        a = np.zeros((7, 7))
        a[1, 0] = 1/5
        a[2, 0] = 3/40;       a[2, 1] = 9/40
        a[3, 0] = 44/45;      a[3, 1] = -56/15;    a[3, 2] = 32/9
        a[4, 0] = 19372/6561; a[4, 1] = -25360/2187; a[4, 2] = 64448/6561; a[4, 3] = -212/729
        a[5, 0] = 9017/3168;  a[5, 1] = -355/33;   a[5, 2] = 46732/5247; a[5, 3] = 49/176;   a[5, 4] = -5103/18656
        a[6, 0] = 35/384;     a[6, 1] = 0;         a[6, 2] = 500/1113;   a[6, 3] = 125/192;  a[6, 4] = -2187/6784;  a[6, 5] = 11/84
        
        b  = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        bs = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        
    return a, b, bs, c, order  


def rk_step(f, t, y, h, a, b, bs, c):
    # f: The physics function f(t, y)
    # t, y: Current time and state
    # h: The step size (dt)
    # a, b, bs, c: The coefficients from get_tableau
    
    num_stages = len(c)
    k = np.zeros((num_stages, len(y)))
    
    # Calculate slopes (k)
    
    # 3. Iterate through each stage
    for i in range(num_stages):
        y_stage = y + h * np.dot(a[i, :], k)
        k[i, :] = f(t + c[i] * h, y_stage)
        
    # Calculate results
    y_new = y + h * np.dot(b, k)

    #Calculate error for adaptive step size
    error = h * np.dot((b - bs), k)
    
    return y_new, error

def solve_ode(f, t_span, y0, rtol=1e-3, atol=1e-6, method="Bogacki-Shampine"):
    
    t_start, t_end = t_span
    a, b, bs, c, order = get_tableau(method)
    
    t = t_start
    y = np.array(y0, dtype=float)
    
    # Initial step guess (conservative)
    h = 0.1 * (t_end - t_start)
    
    # Output history
    t_history = [t]
    y_history = [y]
    
    while t < t_end:
        # Don't overshoot the end time
        if t + h > t_end:
            h = t_end - t
            
        # Take a step
        y_new, error_vector = rk_step(f, t, y, h, a, b, bs, c)
        
        # Industry Standard Error Check
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        error_ratio = np.linalg.norm(error_vector / scale)
        
        # Accept or Reject?
        if error_ratio < 1.0:
            # --- ACCEPT STEP ---
            t += h
            y = y_new
            t_history.append(t)
            y_history.append(y)
            
            # Increase step size (safety factor 0.9)
            # Avoid divide by zero if error is super small
            if error_ratio == 0: 
                h *= 5.0
            else:
                h *= 0.9 * (error_ratio ** (-1/order))
                
            # Cap growth to 5x
            h = min(h, 5.0 * (t_history[-1] - t_history[-2]))
            
        else:
            # --- REJECT STEP ---
            # Shrink step size and try again
            h *= 0.9 * (error_ratio ** (-1/order))
            
            # Prevent h from vanishing to zero
            if h < 1e-15:
                raise ValueError("Step size too small! Stiff problem?")

    return np.array(t_history), np.array(y_history)