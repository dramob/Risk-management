import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Fonction de résolution d'un système tridiagonal (algorithme de Thomas)
# =============================================================================
def tridiag_solver(a, b, c, d):
    """
    Résout le système tridiagonal:
      a[i] * u[i-1] + b[i] * u[i] + c[i] * u[i+1] = d[i], pour i=0,...,N-1,
    avec a[0]=0 et c[N-1]=0.
    Renvoie le vecteur solution u.
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    u = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    u[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i+1]
    return u

# =============================================================================
# Modèle Black-Scholes (option call européenne)
# =============================================================================
class BlackScholes:
    def __init__(self, K, sigma, r, b, S_min, S_max, Nx, tmin, tmax, Nt, theta=0.5):
        self.K = K
        self.sigma = sigma
        self.r = r
        self.b = b      # coût de portage
        self.S_min = S_min
        self.S_max = S_max
        self.Nx = Nx
        self.tmin = tmin
        self.tmax = tmax
        self.Nt = Nt
        self.theta = theta  # paramètre du schéma (0.5 pour Crank-Nicolson)
        self.dS = (S_max - S_min) / (Nx - 1)
        self.dt = (tmax - tmin) / (Nt - 1)

    def payoff(self, S):
        return np.maximum(S - self.K, 0)

    def left_bc(self, t):
        return 0.0

    def right_bc(self, t):
        return self.S_max - self.K * np.exp(-self.r * (self.tmax - t))

    def solve(self):
        S = np.linspace(self.S_min, self.S_max, self.Nx)
        t = np.linspace(self.tmin, self.tmax, self.Nt)
        U = np.zeros((self.Nt, self.Nx))
        # Condition terminale à t = tmax
        U[-1, :] = self.payoff(S)
        # Conditions aux bords pour tout t
        U[:, 0] = self.left_bc(t)
        U[:, -1] = self.right_bc(t)
        
        # Schéma Crank-Nicolson (backward dans le temps)
        for n in range(self.Nt - 2, -1, -1):
            N_int = self.Nx - 2
            A = np.zeros(N_int)
            B = np.zeros(N_int)
            C = np.zeros(N_int)
            d = np.zeros(N_int)
            for i in range(1, self.Nx - 1):
                xi = S[i]
                # Coefficients de l'opérateur spatial :
                a_val = 0.5 * self.sigma**2 * xi**2
                b_val = self.b * xi
                c_val = self.r
                # Implicite
                A_i = - self.theta * self.dt * (a_val / self.dS**2 - b_val / (2*self.dS))
                B_i = 1 + self.theta * self.dt * (2 * a_val / self.dS**2 + c_val)
                C_i = - self.theta * self.dt * (a_val / self.dS**2 + b_val / (2*self.dS))
                # Explicite
                A_exp = (1 - self.theta) * self.dt * (a_val / self.dS**2 - b_val / (2*self.dS))
                B_exp = 1 - (1 - self.theta) * self.dt * (2 * a_val / self.dS**2 + c_val)
                C_exp = (1 - self.theta) * self.dt * (a_val / self.dS**2 + b_val / (2*self.dS))
                
                idx = i - 1
                A[idx] = A_i
                B[idx] = B_i
                C[idx] = C_i
                d[idx] = A_exp * U[n+1, i-1] + B_exp * U[n+1, i] + C_exp * U[n+1, i+1]
            # Intégration des conditions aux bords dans d
            d[0]   -= A[0] * self.left_bc(t[n])
            d[-1]  -= C[-1] * self.right_bc(t[n])
            U[n, 1:-1] = tridiag_solver(A, B, C, d)
        return S, t, U

    def plot(self, S, t, U, filename=None):
        plt.figure(figsize=(10, 6))
        plt.plot(S, U[0, :], label="Prix de l'option (t=tmin)")
        plt.title("Black-Scholes Option Pricing")
        plt.xlabel("Prix de l'actif sous-jacent")
        plt.ylabel("Prix de l'option")
        plt.grid(True)
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()

# =============================================================================
# Modèle CIR (Cox-Ingersoll-Ross) pour les taux d'intérêt
# =============================================================================
class CIR:
    def __init__(self, kappa, theta_cir, sigma, tmin, tmax, Nt, xmin, xmax, Nx, theta=0.5):
        self.kappa = kappa
        self.theta_cir = theta_cir
        self.sigma = sigma
        self.tmin = tmin
        self.tmax = tmax
        self.Nt = Nt
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.theta = theta
        self.dx = (xmax - xmin) / (Nx - 1)
        self.dt = (tmax - tmin) / (Nt - 1)

    def payoff(self, x):
        # Pour le pricing d'une obligation zéro-coupon, u(T,x)=1
        return 1.0

    def left_bc(self, t):
        return 1.0

    def right_bc(self, t):
        return 1.0

    def solve(self):
        x = np.linspace(self.xmin, self.xmax, self.Nx)
        t = np.linspace(self.tmin, self.tmax, self.Nt)
        U = np.zeros((self.Nt, self.Nx))
        U[-1, :] = self.payoff(x)
        U[:, 0] = self.left_bc(t)
        U[:, -1] = self.right_bc(t)
        
        for n in range(self.Nt - 2, -1, -1):
            N_int = self.Nx - 2
            A = np.zeros(N_int)
            B = np.zeros(N_int)
            C = np.zeros(N_int)
            d = np.zeros(N_int)
            for i in range(1, self.Nx - 1):
                xi = x[i]
                # Coefficients pour le modèle CIR
                a_val = 0.5 * self.sigma**2 * xi  # aProc = 0.5 * sigma^2 * x
                b_val = self.kappa * (self.theta_cir - xi)  # bProc
                c_val = xi  # r(t,x)=x
                # Implicite
                A_i = - self.theta * self.dt * (a_val / self.dx**2 - b_val / (2*self.dx))
                B_i = 1 + self.theta * self.dt * (2*a_val/self.dx**2 + c_val)
                C_i = - self.theta * self.dt * (a_val / self.dx**2 + b_val / (2*self.dx))
                # Explicite
                A_exp = (1 - self.theta) * self.dt * (a_val / self.dx**2 - b_val / (2*self.dx))
                B_exp = 1 - (1 - self.theta) * self.dt * (2*a_val/self.dx**2 + c_val)
                C_exp = (1 - self.theta) * self.dt * (a_val / self.dx**2 + b_val / (2*self.dx))
                idx = i - 1
                A[idx] = A_i
                B[idx] = B_i
                C[idx] = C_i
                d[idx] = A_exp * U[n+1, i-1] + B_exp * U[n+1, i] + C_exp * U[n+1, i+1]
            d[0]   -= A[0] * self.left_bc(t[n])
            d[-1]  -= C[-1] * self.right_bc(t[n])
            U[n, 1:-1] = tridiag_solver(A, B, C, d)
        return x, t, U

    def plot(self, x, t, U, filename=None):
        plt.figure(figsize=(10, 6))
        plt.plot(x, U[0, :], label="Solution à t=tmin")
        plt.title("CIR Model")
        plt.xlabel("Taux d'intérêt")
        plt.ylabel("Valeur")
        plt.grid(True)
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()

# =============================================================================
# Modèle Merton (option pricing avec sauts)
# =============================================================================
class Merton:
    def __init__(self, K, sigma, r, b, S_min, S_max, Nx, tmin, tmax, Nt,
                 lambda_, mu_jump, sigma_jump, theta=0.5):
        self.K = K
        self.sigma = sigma
        self.r = r
        self.b = b
        self.S_min = S_min
        self.S_max = S_max
        self.Nx = Nx
        self.tmin = tmin
        self.tmax = tmax
        self.Nt = Nt
        self.lambda_ = lambda_
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.theta = theta
        self.dS = (S_max - S_min) / (Nx - 1)
        self.dt = (tmax - tmin) / (Nt - 1)

    def payoff(self, S):
        return np.maximum(S - self.K, 0)

    def left_bc(self, t):
        return 0.0

    def right_bc(self, t):
        return self.S_max - self.K * np.exp(-self.r * (self.tmax - t))

    def solve(self):
        S = np.linspace(self.S_min, self.S_max, self.Nx)
        t = np.linspace(self.tmin, self.tmax, self.Nt)
        U = np.zeros((self.Nt, self.Nx))
        U[-1, :] = self.payoff(S)
        U[:, 0] = self.left_bc(t)
        U[:, -1] = self.right_bc(t)
        
        for n in range(self.Nt - 2, -1, -1):
            N_int = self.Nx - 2
            A = np.zeros(N_int)
            B = np.zeros(N_int)
            C = np.zeros(N_int)
            d = np.zeros(N_int)
            for i in range(1, self.Nx - 1):
                xi = S[i]
                # Coefficients pour Merton
                jump_term = self.lambda_ * (np.exp(self.mu_jump + 0.5*self.sigma_jump**2) - 1)
                a_val = 0.5 * self.sigma**2 * xi**2
                b_val = - self.b * xi
                c_val = self.r
                A_i = - self.theta * self.dt * (a_val/self.dS**2 - b_val/(2*self.dS))
                B_i = 1 + self.theta * self.dt * (2*a_val/self.dS**2 + c_val + jump_term)
                C_i = - self.theta * self.dt * (a_val/self.dS**2 + b_val/(2*self.dS))
                A_exp = (1 - self.theta) * self.dt * (a_val/self.dS**2 - b_val/(2*self.dS))
                B_exp = 1 - (1 - self.theta) * self.dt * (2*a_val/self.dS**2 + c_val + jump_term)
                C_exp = (1 - self.theta) * self.dt * (a_val/self.dS**2 + b_val/(2*self.dS))
                idx = i - 1
                A[idx] = A_i
                B[idx] = B_i
                C[idx] = C_i
                d[idx] = A_exp * U[n+1, i-1] + B_exp * U[n+1, i] + C_exp * U[n+1, i+1]
            d[0]   -= A[0] * self.left_bc(t[n])
            d[-1]  -= C[-1] * self.right_bc(t[n])
            U[n, 1:-1] = tridiag_solver(A, B, C, d)
        return S, t, U

    def plot(self, S, t, U, filename=None):
        plt.figure(figsize=(10, 6))
        plt.plot(S, U[0, :], label="Prix de l'option à t=tmin")
        plt.title("Merton Model Option Pricing (Crank-Nicolson)")
        plt.xlabel("Prix de l'action")
        plt.ylabel("Prix de l'option")
        plt.grid(True)
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()

# =============================================================================
# Sauvegarde des résultats
# =============================================================================
def save_results(grid, t, U, filename):
    df = pd.DataFrame(U, index=t, columns=grid)
    df.to_csv(filename)

# =============================================================================
# Fonction principale
# =============================================================================
def main():
    # Modèle Black-Scholes
    bs_params = {
        'K': 100,
        'sigma': 0.20,
        'r': 0.08,
        'b': -0.04,
        'S_min': 50,
        'S_max': 150,
        'Nx': 201,
        'tmin': 0,
        'tmax': 0.25,
        'Nt': 1000,
        'theta': 0.5
    }
    bs_model = BlackScholes(**bs_params)
    S_bs, t_bs, U_bs = bs_model.solve()
    bs_model.plot(S_bs, t_bs, U_bs, filename='black_scholes_plot.png')
    save_results(S_bs, t_bs, U_bs, 'black_scholes_results.csv')
    
    # Modèle CIR
    cir_params = {
        'kappa': 0.8,
        'theta_cir': 0.10,
        'sigma': 0.5,
        'tmin': 0,
        'tmax': 5,
        'Nt': 101,
        'xmin': 0,
        'xmax': 1,
        'Nx': 51,
        'theta': 0.5
    }
    cir_model = CIR(**cir_params)
    x_cir, t_cir, U_cir = cir_model.solve()
    cir_model.plot(x_cir, t_cir, U_cir, filename='cir_plot.png')
    save_results(x_cir, t_cir, U_cir, 'cir_results.csv')
    
    # Modèle Merton
    merton_params = {
        'K': 100,
        'sigma': 0.20,
        'r': 0.08,
        'b': -0.04,
        'S_min': 50,
        'S_max': 150,
        'Nx': 201,
        'tmin': 0,
        'tmax': 0.25,
        'Nt': 2000,  # augmentation de Nt pour la stabilité
        'lambda_': 0.1,
        'mu_jump': -0.1,
        'sigma_jump': 0.1,
        'theta': 0.5
    }
    merton_model = Merton(**merton_params)
    S_m, t_m, U_m = merton_model.solve()
    merton_model.plot(S_m, t_m, U_m, filename='merton_plot.png')
    save_results(S_m, t_m, U_m, 'merton_results.csv')

if __name__ == "__main__":
    main()