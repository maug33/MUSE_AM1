def estimate_richardson_error(stepper_func, f, u0, t_span, h, p, **kwargs):
    """
    Estimates error via Richardson Extrapolation by comparing step h and h/2.

    Args:
        p: Theoretical order of the method (1 for Euler, 4 for RK4, etc.)
    Returns:
        u_fine: The solution at t_end using step h/2
        error: Estimated local error vector
    """
    t_start, t_end = t_span

    # Solution with step h
    traj_coarse, _ = integrate(stepper_func, u0, t_start, t_end, f, h, **kwargs)
    u_coarse = traj_coarse[-1]

    # Solution with step h/2
    traj_fine, _ = integrate(stepper_func, u0, t_start, t_end, f, h/2.0, **kwargs)
    u_fine = traj_fine[-1]

    # Richardson error formula
    error_estimate = (u_fine - u_coarse) / (2**p - 1)

    return u_fine, error_estimate

def evaluate_convergence_rate(stepper_func, f, u0, t_span, u_exact_func, p_theoretical, **kwargs):
    """
    Calculates and prints the empirical order of convergence 'p'.
    Requires a known exact solution function u_exact_func(t).
    """
    import matplotlib.pyplot as plt

    t_start, t_end = t_span
    h_values = np.array([0.1, 0.05, 0.025, 0.01]) # Decreasing steps
    errors = []

    u_true = u_exact_func(t_end)

    for h in h_values:
        traj, t = integrate(stepper_func, u0, t_start, t_end, f, h, **kwargs)
        u_num = traj[-1]
        # L2 norm of the error vector
        err = np.linalg.norm(u_num - u_true)
        errors.append(err)

    # Linear regression on log-log data: log(E) = p * log(h) + C
    log_h = np.log(h_values)
    log_E = np.log(errors)
    coeffs = np.polyfit(log_h, log_E, 1)
    p_measured = coeffs[0]

    print(f"{stepper_func.__name__}: Theoretical p={p_theoretical}, Measured p={p_measured:.4f}")

    return p_measured, h_values, errors