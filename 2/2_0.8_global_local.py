import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

def d_xi(x, y, nabla_xi):
    dxdxi = nabla_xi @ x
    dydxi = nabla_xi @ y
    return dxdxi, dydxi

def d_eta(x, y, nabla_eta):
    dxdeta = x @ nabla_eta
    dydeta = y @ nabla_eta
    return dxdeta, dydeta

def d_xi_eta(dxdeta, dydeta, nabla_xi):
    dxdxideta = nabla_xi @ dxdeta
    dydxideta = nabla_xi @ dydeta
    return dxdxideta, dydxideta

def d_eta_eta(x, y, laplace_eta):
    dxdetadeta = x @ laplace_eta
    dydetadeta = y @ laplace_eta
    return dxdetadeta, dydetadeta

def d_xi_xi(x, y, laplace_xi):
    dxdxidxi = laplace_xi @ x
    dydxidxi = laplace_xi @ y
    return dxdxidxi, dydxidxi

def FluParaCalFromU(U, gamma):
    rho = U[0, :, :]
    u = U[1, :, :] / rho
    v = U[2, :, :] / rho
    E = U[3, :, :] / rho
    e = E - 0.5 * (u**2 + v**2)
    p = (gamma - 1) * rho * e
    return rho, u, v, p, E

def FVS_cal(rho, u, v, E, p, J, zeta_x, zeta_y, a, gamma):
    gridSize1, gridSize2 = J.shape
    grad_zeta = np.sqrt(zeta_x**2 + zeta_y**2)
    zeta_x_tilde = zeta_x / grad_zeta
    zeta_y_tilde = zeta_y / grad_zeta
    theta_tidle = zeta_x_tilde * u + zeta_y_tilde * v
    lam1 = zeta_x * u + zeta_y * v - a * grad_zeta
    lam2 = zeta_x * u + zeta_y * v
    lam4 = zeta_x * u + zeta_y * v + a * grad_zeta
    lam1_posi = 0.5 * (lam1 + np.abs(lam1))
    lam2_posi = 0.5 * (lam2 + np.abs(lam2))
    lam4_posi = 0.5 * (lam4 + np.abs(lam4))
    lam1_nega = 0.5 * (lam1 - np.abs(lam1))
    lam2_nega = 0.5 * (lam2 - np.abs(lam2))
    lam4_nega = 0.5 * (lam4 - np.abs(lam4))

    H1_hat = np.zeros((4, gridSize1, gridSize2))
    H2_hat = np.zeros((4, gridSize1, gridSize2))
    H4_hat = np.zeros((4, gridSize1, gridSize2))

    H1_hat[0, :, :] = rho
    H1_hat[1, :, :] = rho * u - rho * a * zeta_x_tilde
    H1_hat[2, :, :] = rho * v - rho * a * zeta_y_tilde
    H1_hat[3, :, :] = rho * E + p - rho * a * theta_tidle
    H1_hat = 1 / (2 * gamma) * J * H1_hat

    H2_hat[0, :, :] = rho
    H2_hat[1, :, :] = rho * u
    H2_hat[2, :, :] = rho * v
    H2_hat[3, :, :] = 0.5 * rho * (u**2 + v**2)
    H2_hat = (gamma - 1) / gamma * J * H2_hat

    H4_hat[0, :, :] = rho
    H4_hat[1, :, :] = rho * u + rho * a * zeta_x_tilde
    H4_hat[2, :, :] = rho * v + rho * a * zeta_y_tilde
    H4_hat[3, :, :] = rho * E + p + rho * a * theta_tidle
    H4_hat = 1 / (2 * gamma) * J * H4_hat

    H_posi = lam1_posi * H1_hat + lam2_posi * H2_hat + lam4_posi * H4_hat
    H_nega = lam1_nega * H1_hat + lam2_nega * H2_hat + lam4_nega * H4_hat

    return H_posi, H_nega

def innerBoundary(W_in_star, gamma):
    size_W = W_in_star.shape
    M_xi = size_W[1] - 1
    xi = np.arange(1, M_xi + 2)
    theta = ((xi - 1) * 2 * np.pi / M_xi).reshape(-1, 1)
    rho_star = W_in_star[0, :].reshape(-1, 1)
    u_star = W_in_star[1, :].reshape(-1, 1)
    v_star = W_in_star[2, :].reshape(-1, 1)
    p_star = W_in_star[3, :].reshape(-1, 1)
    a_star = np.sqrt(np.abs(gamma * p_star / rho_star))  # Avoid invalid values
    Vn_star = u_star * np.cos(theta) - v_star * np.sin(theta)

    u = u_star * (-np.sin(theta)) * (-np.sin(theta)) - v_star * (-np.sin(theta)) * np.cos(theta)
    v = -u_star * (-np.sin(theta)) * np.cos(theta) + v_star * np.cos(theta) * np.cos(theta)
    rho = ((gamma - 1)**2 * (Vn_star - 2 * a_star / (gamma - 1))**2 * (np.abs(rho_star))**gamma / (4 * gamma * np.abs(p_star)))**(1 / (gamma - 1))
    p = p_star * (rho / np.abs(rho_star))**gamma
    W = np.vstack((rho.T, u.T, v.T, p.T))
    return W

def outerBoundary(W_out_star, gamma, W_infty):
    size_W = W_out_star.shape
    M_xi = size_W[1] - 1
    xi = np.arange(1, M_xi + 2)
    theta = ((xi - 1) * 2 * np.pi / M_xi).reshape(-1, 1)
    rho_star = W_out_star[0, :].reshape(-1, 1)
    u_star = W_out_star[1, :].reshape(-1, 1)
    v_star = W_out_star[2, :].reshape(-1, 1)
    p_star = W_out_star[3, :].reshape(-1, 1)
    a_star = np.sqrt(np.abs(gamma * p_star / rho_star))  # Avoid invalid values
    Vn_star = u_star * np.cos(theta) - v_star * np.sin(theta)
    Vt_star = -v_star * np.cos(theta) - u_star * np.sin(theta)

    rho_infty = W_infty[0, :].reshape(-1, 1)
    u_infty = W_infty[1, :].reshape(-1, 1)
    v_infty = W_infty[2, :].reshape(-1, 1)
    p_infty = W_infty[3, :].reshape(-1, 1)
    a_infty = np.sqrt(np.abs(gamma * p_infty / rho_infty))  # Avoid invalid values
    Vn_infty = u_infty * np.cos(theta) - v_infty * np.sin(theta)
    Vt_infty = -v_infty * np.cos(theta) - u_infty * np.sin(theta)

    Vn = 0.5 * (Vn_infty + Vn_star + 2 * a_star / (gamma - 1) - 2 * a_infty / (gamma - 1))
    Vt = (Vn >= 0) * Vt_star + (Vn < 0) * Vt_infty
    u = Vn * np.cos(theta) - Vt * np.sin(theta)
    v = -Vn * np.sin(theta) - Vt * np.cos(theta)
    a = (gamma - 1) / 4 * (Vn_star + 2 * a_star / (gamma - 1) - Vn_infty + 2 * a_infty / (gamma - 1))
    rho = (Vn >= 0) * ((gamma / a / a * p_star / (np.abs(rho_star)**gamma))**(1 / (1 - gamma))) + (Vn < 0) * ((gamma / a / a * p_infty / (np.abs(rho_infty)**gamma))**(1 / (1 - gamma)))

    p = a**2 / gamma * rho

    W = np.vstack((rho.T, u.T, v.T, p.T))
    return W

def run_simulation(use_local_time_step):
    # Load grid data
    data = np.load('grid_circle.npz')
    print("Keys in the .npz file:", data.keys())

    x_key = 'X' 
    y_key = 'Y' 

    if x_key in data and y_key in data:
        x = data[x_key]
        y = data[y_key]

        # Flip xi direction
        x = np.flipud(x)
        y = np.flipud(y)
        grid_size = x.shape

        # Grid points and mesh size
        M_xi = grid_size[0] - 1
        M_eta = grid_size[1] - 1
        delta_xi = 1
        delta_eta = 1
        xi = np.arange(1, M_xi + 2)
        eta = np.arange(1, M_eta + 2)
        theta = ((xi - 1) * 2 * np.pi / M_xi).reshape(-1, 1)

        # Time step and CFL condition
        dt_cal = 0.00005
        CFL_max = 0.005

        # Physical parameters
        gamma = 1.4
        Ma = 0.8
        rho0 = np.ones_like(x)
        p0 = np.ones_like(x)
        a = np.sqrt(np.abs(gamma * p0 / rho0))  # Speed of sound

        u0 = Ma * a
        v0 = np.zeros_like(x)
        e0 = p0 / rho0 / (gamma - 1)
        E0 = e0 + 0.5 * (u0**2 + v0**2)
        Ber0 = 0.5 * (u0**2 + v0**2) + p0 / rho0  # Bernoulli constant

        # Differential matrices
        nabla_xi = spdiags([-1 * np.ones(M_xi + 1), np.zeros(M_xi + 1), np.ones(M_xi + 1)], [-1, 0, 1], M_xi + 1, M_xi + 1).toarray() / 2
        nabla_eta = spdiags([np.ones(M_eta + 1), np.zeros(M_eta + 1), -1 * np.ones(M_eta + 1)], [-1, 0, 1], M_eta + 1, M_eta + 1).toarray() / 2
        laplace_xi = spdiags([np.ones(M_xi + 1), -2 * np.ones(M_xi + 1), np.ones(M_xi + 1)], [-1, 0, 1], M_xi + 1, M_xi + 1).toarray()
        laplace_eta = spdiags([np.ones(M_eta + 1), -2 * np.ones(M_eta + 1), np.ones(M_eta + 1)], [-1, 0, 1], M_eta + 1, M_eta + 1).toarray()

        # Boundary conditions
        nabla_xi[0, M_xi] = -0.5
        nabla_xi[M_xi, 1] = 0.5  # Periodic boundary
        nabla_eta[0, 0] = -1
        nabla_eta[1, 0] = 1  # Solid wall boundary
        nabla_eta[M_eta - 1, M_eta] = -1
        nabla_eta[M_eta, M_eta] = 1  # Far-field boundary
        laplace_xi[0, M_xi] = 1
        laplace_xi[M_xi, 1] = 1
        laplace_eta[:, 0] = 0
        laplace_eta[:, M_eta] = 0  # Solid wall boundary

        # Calculate derivatives
        x_xi, y_xi = d_xi(x, y, nabla_xi)
        x_eta, y_eta = d_eta(x, y, nabla_eta)
        x_xi_eta, y_xi_eta = d_xi_eta(x_eta, y_eta, nabla_xi)
        x_xi_xi, y_xi_xi = d_xi_xi(x, y, laplace_xi)
        x_eta_eta, y_eta_eta = d_eta_eta(x, y, laplace_eta)

        # Jacobian
        J = x_xi * y_eta - x_eta * y_xi
        J_cal = J.reshape(1, M_xi + 1, M_eta + 1)
        x_xi_cal = x_xi.reshape(1, M_xi + 1, M_eta + 1)
        x_eta_cal = x_eta.reshape(1, M_xi + 1, M_eta + 1)
        y_xi_cal = y_xi.reshape(1, M_xi + 1, M_eta + 1)
        y_eta_cal = y_eta.reshape(1, M_xi + 1, M_eta + 1)
        xi_x = y_eta / J
        xi_y = -x_eta / J
        eta_x = -y_xi / J
        eta_y = x_xi / J

        rho = rho0.copy()
        u = u0.copy()
        v = v0.copy()
        e = e0.copy()
        E = E0.copy()
        p = p0.copy()
        Ber = 0.5 * (u**2 + v**2) + p / rho

        W_infty = np.vstack((rho0[:, M_eta].T, Ma * a[:, M_eta].T, np.zeros(a[:, M_eta].T.shape), p0[:, M_eta].T))
        W_in_star = 2 * np.vstack((rho[:, 1].T, u[:, 1].T, v[:, 1].T, p[:, 1].T)) - np.vstack((rho[:, 2].T, u[:, 2].T, v[:, 2].T, p[:, 2].T))
        W_out_star = 2 * np.vstack((rho[:, M_eta].T, u[:, M_eta].T, v[:, M_eta].T, p[:, M_eta].T)) - np.vstack((rho[:, M_eta - 1].T, u[:, M_eta - 1].T, v[:, M_eta - 1].T, p[:, M_eta - 1].T))

        W_inner = innerBoundary(W_in_star, gamma)
        W_outer = outerBoundary(W_out_star, gamma, W_infty)
        rho[:, 0] = W_inner[0, :].T
        u[:, 0] = W_inner[1, :].T
        v[:, 0] = W_inner[2, :].T
        p[:, 0] = W_inner[3, :].T
        rho[:, M_eta] = W_outer[0, :].T
        u[:, M_eta] = W_outer[1, :].T
        v[:, M_eta] = W_outer[2, :].T
        p[:, M_eta] = W_outer[3, :].T
        a = np.sqrt(np.abs(gamma * p / rho))
        e = p / rho / (gamma - 1)
        E = e + 0.5 * (u**2 + v**2)

        # Convert to U_hat, F_hat, G_hat
        U = np.zeros((4, M_xi + 1, M_eta + 1))
        U[0, :, :] = rho
        U[1, :, :] = rho * u
        U[2, :, :] = rho * v
        U[3, :, :] = rho * E

        U_hat = J_cal * U

        F_posi, F_nega = FVS_cal(rho, u, v, E, p, J, xi_x, xi_y, a, gamma)
        G_posi, G_nega = FVS_cal(rho, u, v, E, p, J, eta_x, eta_y, a, gamma)

        # Iteration parameters
        iterlimit = 100000
        errlimit = 1e-11
        errupbound = 5
        rec_max_err = np.zeros(iterlimit)
        P = np.zeros((4, M_xi + 1, M_eta + 1))

        # Local time step calculation
        lambda_xi_max = np.abs(xi_x * u + xi_y * v) + a * np.sqrt(xi_x**2 + xi_y**2)
        lambda_eta_max = np.abs(eta_x * u + eta_y * v) + a * np.sqrt(eta_x**2 + eta_y**2)
        dt_local = CFL_max / (lambda_xi_max + lambda_eta_max)
        
        # Global time step
        dt_global = np.min(dt_local)

        # Choose time stepping method
        dt = dt_local if use_local_time_step else dt_global

        for iter in range(iterlimit):
            F_nega = np.real(F_nega)
            F_posi = np.real(F_posi)
            G_nega = np.real(G_nega)
            G_posi = np.real(G_posi)
            U_hat = np.real(U_hat)
            U = np.real(U)
            
            U_hat_hold = U_hat.copy()
            U1_hat = np.zeros_like(U_hat)  # Initialize U1_hat here
            for R in range(4):
                for j in range(1, M_eta):
                    f_posi = F_posi[:, :, j]
                    f_nega = F_nega[:, :, j]
                    P_f = (f_posi - np.hstack((f_posi[:, M_xi].reshape(-1, 1), f_posi[:, :M_xi]))) / delta_xi + (np.hstack((f_nega[:, 1:], f_nega[:, 1].reshape(-1, 1))) - f_nega) / delta_xi
                    
                    g_ij_posi = G_posi[:, :, j]
                    g_ijm_posi = G_posi[:, :, j - 1]
                    g_ij_nega = G_nega[:, :, j]
                    g_ijp_nega = G_nega[:, :, j + 1]
                    P_g = (g_ij_posi - g_ijm_posi) / delta_eta + (g_ijp_nega - g_ij_nega) / delta_eta
                    
                    P[:, :, j] = P_f + P_g

                if use_local_time_step:
                    for i in range(M_xi + 1):
                        for j in range(M_eta + 1):
                            U1_hat[:, i, j] = U_hat_hold[:, i, j] - dt_local[i, j] / (5 - R) * P[:, i, j]
                else:
                    U1_hat = U_hat_hold - dt_global / (5 - R) * P
                    
                U1 = U1_hat / J_cal
                rho, u, v, p, E = FluParaCalFromU(U1, gamma)
                
                W_in_star = 2 * np.vstack((rho[:, 1].T, u[:, 1].T, v[:, 1].T, p[:, 1].T)) - np.vstack((rho[:, 2].T, u[:, 2].T, v[:, 2].T, p[:, 2].T))
                W_out_star = 2 * np.vstack((rho[:, M_eta].T, u[:, M_eta].T, v[:, M_eta].T, p[:, M_eta].T)) - np.vstack((rho[:, M_eta - 1].T, u[:, M_eta - 1].T, v[:, M_eta - 1].T, p[:, M_eta - 1].T))
                
                W_inner = innerBoundary(W_in_star, gamma)
                W_outer = outerBoundary(W_out_star, gamma, W_infty)
                
                rho[:, 0] = W_inner[0, :].T
                u[:, 0] = W_inner[1, :].T
                v[:, 0] = W_inner[2, :].T
                p[:, 0] = W_inner[3, :].T
                rho[:, M_eta] = W_outer[0, :].T
                u[:, M_eta] = W_outer[1, :].T
                v[:, M_eta] = W_outer[2, :].T
                p[:, M_eta] = W_outer[3, :].T
                
                a = np.sqrt(np.abs(gamma * p / rho))
                e = p / rho / (gamma - 1)
                E = e + 0.5 * (u**2 + v**2)
                
                F_posi, F_nega = FVS_cal(rho, u, v, E, p, J, xi_x, xi_y, a, gamma)
                G_posi, G_nega = FVS_cal(rho, u, v, E, p, J, eta_x, eta_y, a, gamma)
                
                Ber = 0.5 * (u**2 + v**2) + p / rho

            U_hat = U1_hat
            max_err = np.max(np.abs(U_hat_hold[1, :, :] / U_hat_hold[0, :, :] - U1_hat[1, :, :] / U1_hat[0, :, :]))
            rec_max_err[iter] = max_err
            print(f'iter = {iter}, max_err = {max_err}')
            
            if max_err < errlimit:
                break
            if max_err > errupbound:
                print('Error exceeds upper bound.')
                break

        return rec_max_err[:iter]

    else:
        print(f"Keys '{x_key}' and/or '{y_key}' not found in the file.")
        return None

# Run simulations
err_local_time_step = run_simulation(use_local_time_step=True)
err_global_time_step = run_simulation(use_local_time_step=False)

# Plotting
plt.plot(np.arange(len(err_local_time_step)), err_local_time_step, label='Local Time Step')
plt.plot(np.arange(len(err_global_time_step)), err_global_time_step, label='Global Time Step')
plt.title('Error Convergence Comparison')
plt.xlabel('Iteration')
plt.ylabel('Max Error')
plt.legend()
plt.savefig('error_convergence_comparison.png')
plt.show()
