import numpy as np
from scipy.constants import c, mu_0, epsilon_0

def divergence(Fx, Fy, dx, dy):
    """∇·F on a 2D staggered grid."""
    dFx_dx = (np.roll(Fx, -1, axis=0) - np.roll(Fx, 1, axis=0)) / (2*dx)
    dFy_dy = (np.roll(Fy, -1, axis=1) - np.roll(Fy, 1, axis=1)) / (2*dy)
    return dFx_dx + dFy_dy

def curl_E_to_dB(Ez, dx, dy):
    """Compute -∂B/∂t = ∇×E for TMz mode (Ez component).
       Returns dBx_dt and dBy_dt on the B-grid."""
    dEz_dy = (np.roll(Ez, -1, axis=0) - np.roll(Ez, 1, axis=0)) / (2*dy)
    dEz_dx = (np.roll(Ez, -1, axis=1) - np.roll(Ez, 1, axis=1)) / (2*dx)
    # According to Faraday: dBx/dt = -dEz/dy, dBy/dt = dEz/dx
    return -dEz_dy, dEz_dx

def curl_B_to_dE(Bx, By, dx, dy, mu0, eps0):
    """Compute dE/dt = (1/μ0ε0) ∇×B for TMz mode."""
    dBy_dx = (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / (2*dx)
    dBx_dy = (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / (2*dy)
    # dBz_term not needed for TMz
    # Ampère: curl(B) = μ0 ε0 ∂E/∂t  => ∂E/∂t = (1/μ0ε0) * (dBy_dx - dBx_dy)
    return (dBy_dx - dBx_dy) / (mu0 * eps0)



class MaxwellWiFiChannel:
    """
    2D-FDTD (TMz) simulation of a WiFi router antenna emitting EM waves.
    The modulated signal is fed as a current source at the antenna location.
    The electric field Ez(x_rx, y_rx) at the brain-receiver antenna is recorded.
    """
    def __init__(self, grid_size=(100, 200), dh=0.01, dt=None,
                 src_pos=(25, 30), rx_pos=(75, 120),
                 freq=2.4e9, ppw=20):
        self.nx, self.ny = grid_size
        self.dh = dh  # cell size in metres
        self.dt = dt if dt else dh / (2*c)  # Courant limit
        self.src_x, self.src_y = src_pos
        self.rx_x, self.rx_y = rx_pos

        # Material: free space
        self.eps_r = 1.0
        self.mu_r = 1.0
        self.eps = epsilon_0 * self.eps_r
        self.mu = mu_0 * self.mu_r

        # Fields
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

        # Pulse shape
        self.freq = freq
        self.ppw = ppw  # points per wavelength (for visualisation)
        # Ricker wavelet parameters
        self.tau = 1.0 / freq * 3
        self.t0 = 4 * self.tau

    def update_source(self, t, signal):
        """Inject the modulated signal current at the antenna location."""
        # signal is a single sample value (real) at time step t
        self.Ez[self.src_x, self.src_y] += signal

    def step(self):
        """One FDTD time step (update H then E)."""
        # Update H from E: curl of E
        # Hx at (i,j+1/2), Hy at (i+1/2,j)
        # dEz/dy -> Hx, dEz/dx -> -Hy
        dEz_dy = (self.Ez[1:, 1:] - self.Ez[:-1, 1:]) / self.dh
        self.Hx[:-1, :-1] -= self.dt / self.mu * dEz_dy

        dEz_dx = (self.Ez[1:, 1:] - self.Ez[1:, :-1]) / self.dh
        self.Hy[:-1, :-1] += self.dt / self.mu * dEz_dx

        # Update E from H: curl of H
        # dHy/dx - dHx/dy -> Ez
        dHy_dx = (self.Hy[:-1, 1:] - self.Hy[:-1, :-1]) / self.dh
        dHx_dy = (self.Hx[1:, :-1] - self.Hx[:-1, :-1]) / self.dh
        self.Ez[1:-1, 1:-1] += self.dt / self.eps * (dHy_dx - dHx_dy)

        # Absorbing boundaries (simple Mur first-order, omitted for brevity)

    def simulate(self, samples, signal_func):
        """
        Run simulation for len(samples) time steps,
        feeding signal_func(time_step) as source.
        Returns recorded E-field at receiver location.
        """
        record = np.zeros(len(samples))
        for n in range(len(samples)):
            # Inject source
            src_val = signal_func(n)
            self.update_source(n * self.dt, src_val)
            self.step()
            record[n] = self.Ez[self.rx_x, self.rx_y]
        return record
    
    def modulated_waveform(symbols, fc=2.4e9, fs=20e9):
       """Convert symbol stream to a time-domain signal at carrier freq.
       For BPSK: symbol ∈ {+1,-1} -> cos(2π fc t + phase)"""
       t = np.arange(len(symbols)) / fs
       phase = np.where(symbols.real > 0, 0, np.pi)
       return np.cos(2*np.pi*fc*t + phase)
    
    def channel(self, symbols):
    # create time-domain RF signal
     return rx_iq