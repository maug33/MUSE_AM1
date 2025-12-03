import numpy as np

# ==========================================
# 1. KEPLER PROBLEM (Two-Body)
# ==========================================

def F_kepler(u, t, mu=1.0, epsilon=1e-12):
    """
    Defines the differential equations for a simple Keplerian orbit.

    Args:
        u: State vector [x, y, vx, vy]
        t: Time (unused here but required by solver signature)
        mu: Gravitational parameter (GM)
        epsilon: Softening factor to prevent division by zero

    Returns:
        Derivative vector [vx, vy, ax, ay]
    """
    x, y, vx, vy = u

    # Calculate radius (with softening for numerical safety)
    r_squared = x**2 + y**2 + epsilon**2
    r = np.sqrt(r_squared)
    r3 = r**3

    # Acceleration F = -mu * r / |r|^3
    ax = -mu * x / r3
    ay = -mu * y / r3

    return np.array([vx, vy, ax, ay])


# ==========================================
# 2. HARMONIC OSCILLATOR
# ==========================================

def F_oscillator(u, t):
    """
    Simple Harmonic Oscillator: x'' = -x
    Used for Milestone 3 convergence testing because exact solution is known.
    """
    x, v = u
    return np.array([v, -x])

def u_exact_oscillator(t, u0=None):
    """
    Exact solution for Harmonic Oscillator.
    Assumes default u0=[1, 0] if not provided.
    """
    if u0 is None:
        u0 = np.array([1.0, 0.0])

    x0, v0 = u0
    # x(t) = x0*cos(t) + v0*sin(t)
    x_t = x0 * np.cos(t) + v0 * np.sin(t)
    # v(t) = -x0*sin(t) + v0*cos(t)
    v_t = -x0 * np.sin(t) + v0 * np.cos(t)

    return np.array([x_t, v_t])


# ==========================================
# 3. KEPLER PROBLEM (N-body)
# ==========================================

def F_nbody(u, t, masses, G=1.0, epsilon=1e-12):
    """
    Computes derivatives for the N-body problem.
    
    Args:
        u: State vector [x1, y1, z1, ..., vx1, vy1, vz1, ...]
           Length is 2 * 3 * N.
        masses: List or array of masses for the N bodies.
    """
    N = len(masses)
    n_dims = 3 
    
    # Extract positions and velocities
    mid = N * n_dims
    r_flat = u[:mid]
    v_flat = u[mid:]
    
    # Reshape positions for easier distance calculation
    r = r_flat.reshape((N, n_dims))
    a = np.zeros_like(r)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_vec = r[j] - r[i]
                dist_mag = np.linalg.norm(dist_vec) + epsilon # Softening
                # Acceleration: a_i = sum( G * m_j * vec_r / r^3 )
                a[i] += G * masses[j] * dist_vec / (dist_mag**3)
    
    # Return [velocities, accelerations] flattened
    return np.concatenate((v_flat, a.flatten()))


# ==========================================
# 4. CIRCULAR RESTRICTED 3-BODY PROBLEM (CR3BP)
# ==========================================

def cr3bp_equations(t, u, mu=0.0121505856, **kwargs):
    """
    Defines the equations of motion for the Circular Restricted Three-Body Problem.
    Works in the rotating reference frame.

    Args:
        t: Time (unused in autonomous system but required by solver)
        u: State vector [x, y, vx, vy]
        mu: Mass parameter (ratio m2 / (m1 + m2)). Default is Earth-Moon.
        **kwargs: Absorb extra arguments safely.

    Returns:
        Derivative vector [vx, vy, ax, ay]
    """
    x, y, vx, vy = u
    
    # Distance to Body 1 (Larger Mass at -mu)
    r1 = np.sqrt((x + mu)**2 + y**2)
    
    # Distance to Body 2 (Smaller Mass at 1-mu)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    
    # Effective Potential Gradients (Acceleration terms from Gravity)
    # Omega_x = x - (1-mu)(x+mu)/r1^3 - mu(x-1+mu)/r2^3
    omega_x = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    
    # Omega_y = y - (1-mu)y/r1^3 - mu*y/r2^3
    omega_y = y - (1 - mu) * y / r1**3 - mu * y / r2**3
    
    # Equations of Motion (including Coriolis forces: 2*vy and -2*vx)
    # dx/dt = vx
    dxdt = vx
    
    # dy/dt = vy
    dydt = vy
    
    # dvx/dt = 2*vy + Omega_x
    dvxdt = 2 * vy + omega_x
    
    # dvy/dt = -2*vx + Omega_y
    dvydt = -2 * vx + omega_y
    
    return np.array([dxdt, dydt, dvxdt, dvydt])


def lagrange_residual(u_pos, mu=0.0121505856, **kwargs):
    """
    Calculates the residual Force (Acceleration) for finding Lagrange points.
    Used by the generic Newton solver.

    Args:
        u_pos: Position vector. Can be 1D [x] (for collinear) or 2D [x, y].
        mu: Mass parameter.
        **kwargs: Passed through from Newton solver.

    Returns:
        Vector of accelerations [ax] or [ax, ay]. Target is 0.
    """
    # Handle 1D input (Collinear points L1, L2, L3 assume y=0)
    if len(u_pos) == 1:
        x = u_pos[0]
        y = 0.0
    else:
        x = u_pos[0]
        y = u_pos[1]
        
    # Construct a full state vector with zero velocity [x, y, 0, 0]
    # We use t=0 because the equilibrium points are static in this frame
    full_state = np.array([x, y, 0.0, 0.0])
    
    # Get derivatives using the main physics function
    derivs = cr3bp_equations(0, full_state, mu=mu)
    
    # Extract accelerations (indices 2 and 3)
    ax = derivs[2]
    ay = derivs[3]
    
    # Return match for input shape
    if len(u_pos) == 1:
        return np.array([ax])
    else:
        return np.array([ax, ay])


def get_cr3bp_stability_matrix(u_full, mu=0.0121505856, **kwargs):
    """
    Helper to wrap the physics function for the generic Jacobian solver.
    This strictly accepts F(U) so it works with get_numerical_jacobian.
    """
    # We define a lambda or local function that freezes time at 0
    # and ensures arguments align with the Jacobian tool.
    def physics_wrapper(u_state, **args):
        return cr3bp_equations(0, u_state, mu=mu, **args)
    
    return physics_wrapper