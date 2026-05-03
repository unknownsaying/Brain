import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# ============================================================
# HARMONIC OSCILLATOR SIMULATOR FOR HARDWARE COMPONENTS
# ============================================================
# Each component modeled as: m*x'' + b*x' + k*x = F(t)
# Where: ω₀ = √(k/m), ζ = b/(2√(mk)), F(t) = F₀*cos(ω_drive*t)

class HardwareOscillator:
    """Models a hardware component as a damped, driven harmonic oscillator."""
    
    def __init__(self, name, mass, stiffness, damping, driving_force, driving_freq, 
                 initial_pos=0, initial_vel=0, color='blue', description=''):
        self.name = name
        self.m = mass          # Mass/Inertia
        self.k = stiffness     # Spring constant (restoring force)
        self.b = damping       # Damping coefficient
        self.F0 = driving_force  # Driving force amplitude
        self.f_drive = driving_freq  # Driving frequency (Hz)
        self.x0 = initial_pos
        self.v0 = initial_vel
        self.color = color
        self.desc = description
        
        # Derived parameters
        self.omega0 = np.sqrt(self.k / self.m) if self.m > 0 else 0
        self.zeta = self.b / (2 * np.sqrt(self.m * self.k)) if self.m * self.k > 0 else 0
        self.omega_drive = 2 * np.pi * self.f_drive
        self.quality_factor = 1 / (2 * self.zeta) if self.zeta > 0 else float('inf')
        
        # State
        self.t = None
        self.x = None
        self.v = None
        
    def characteristics(self):
        """Return key characteristics as a dict."""
        return {
            'Natural Freq (Hz)': self.omega0 / (2*np.pi),
            'Damping Ratio ζ': self.zeta,
            'Quality Factor Q': self.quality_factor,
            'Driving Freq (Hz)': self.f_drive,
            'Mass m': self.m,
            'Stiffness k': self.k,
            'Damping b': self.b
        }
    
    def regime(self):
        """Determine the oscillation regime."""
        if self.zeta < 1:
            return f"Underdamped (ζ={self.zeta:.4f})"
        elif np.isclose(self.zeta, 1, rtol=1e-9):
            return f"Critically Damped"
        else:
            return f"Overdamped (ζ={self.zeta:.4f})"

def harmonic_ode(t, y, oscillator):
    """ODE for damped driven harmonic oscillator: dx/dt = v, dv/dt = -2ζω₀v - ω₀²x + F₀cos(ωt)/m"""
    x, v = y
    osc = oscillator
    a = -2 * osc.zeta * osc.omega0 * v - osc.omega0**2 * x
    if osc.F0 != 0:
        a += (osc.F0 / osc.m) * np.cos(osc.omega_drive * t)
    return [v, a]

# ============================================================
# DEFINE HARDWARE COMPONENTS
# ============================================================

hardware_components = {
    'CPU Load': HardwareOscillator(
        name='CPU Load',
        mass=1.0,                    # Normalized mass
        stiffness=(2*np.pi*2.5e9)**2,  # k = m*ω₀², ω₀ corresponds to ~2.5 GHz clock
        damping=0.1 * 2 * np.sqrt(1.0 * (2*np.pi*2.5e9)**2),  # Light damping
        driving_force=0.2,           # Workload variations
        driving_freq=100e6,          # Modulated at 100 MHz (pipeline variations)
        initial_pos=0.5,             # 50% initial load
        initial_vel=0.0,
        color='#FF6B6B',
        description='Pipeline activity: Oscillates between idle (0) and full load (1) due to branch prediction and pipeline flushes'
    ),
    
    'GPU Temp': HardwareOscillator(
        name='GPU Temperature',
        mass=5.0,                    # Thermal mass (larger = slower response)
        stiffness=5.0 * (2*np.pi*0.5)**2,  # Low natural frequency (thermal time constant)
        damping=0.8 * 2 * np.sqrt(5.0 * 5.0 * (2*np.pi*0.5)**2),  # Heavy damping
        driving_force=10.0,          # Heat generation
        driving_freq=0.2,            # Slow load changes
        initial_pos=40.0,            # Starting at 40°C
        initial_vel=0.0,
        color='#FF8C42',
        description='Thermal dynamics: Temperature changes slowly due to large thermal mass, heavily damped by cooling system'
    ),
    
    'RAM Latency': HardwareOscillator(
        name='RAM Latency',
        mass=1e-12,                  # Tiny effective mass (nanosecond scale)
        stiffness=1e-12 * (2*np.pi*1e6)**2,  # ~1 MHz row cycle
        damping=0.05 * 2 * np.sqrt(1e-12 * 1e-12 * (2*np.pi*1e6)**2),  # Very light damping
        driving_force=0.5e-12,
        driving_freq=1e6,            # Row refresh frequency
        initial_pos=5e-9,            # 5 ns initial latency
        initial_vel=0.0,
        color='#FFD93D',
        description='Memory access: Latency oscillates due to row activation, refresh cycles, and bus arbitration'
    ),
    
    'Disk I/O': HardwareOscillator(
        name='Disk I/O',
        mass=0.01,                   # Mechanical inertia
        stiffness=0.01 * (2*np.pi*120)**2,  # 7200 RPM = 120 Hz
        damping=0.9 * 2 * np.sqrt(0.01 * 0.01 * (2*np.pi*120)**2),  # Near-critical damping
        driving_force=100.0,         # Motor torque
        driving_freq=120.0,          # Rotational frequency
        initial_pos=0.0,
        initial_vel=2*np.pi*120 * 1.0,  # Initial tangential velocity
        color='#6BCB77',
        description='Platter rotation: Mechanical oscillator with rotational inertia, bearings provide damping'
    ),
    
    'RISC-V Clock': HardwareOscillator(
        name='RISC-V Clock',
        mass=1e-20,                  # Electron-scale effective mass
        stiffness=1e-20 * (2*np.pi*1e9)**2,  # 1 GHz crystal
        damping=0.001 * 2 * np.sqrt(1e-20 * 1e-20 * (2*np.pi*1e9)**2),  # Ultra-low damping (high Q)
        driving_force=1.0,
        driving_freq=1e9,            # 1 GHz oscillation
        initial_pos=0.0,
        initial_vel=2*np.pi*1e9 * 0.1,
        color='#4D96FF',
        description='Quartz crystal oscillator: Ultra-stable, very high Q-factor (~10^6), nearly pure sine wave'
    ),
    
    'RLC Circuit': HardwareOscillator(
        name='RLC Circuit',
        mass=1e-3,                   # Inductance L = 1 mH
        stiffness=(1e-3) * (31623)**2,  # k = L*ω₀², ω₀ = 1/√(LC), C=1µF
        damping=100.0,               # Resistance R = 100 Ω
        driving_force=5.0,           # Driving voltage amplitude
        driving_freq=1000.0,         # 1 kHz driving frequency
        initial_pos=0.0,
        initial_vel=0.0,
        color='#9B59B6',
        description='L=1mH, C=1µF, R=100Ω: Classical electromagnetic oscillator, overdamped (ζ≈1.58)'
    )
}

# ============================================================
# SIMULATION AND ANALYSIS
# ============================================================

def simulate_oscillator(osc, duration_factor=10, points=5000):
    """Simulate an oscillator over several natural periods."""
    if osc.omega0 == 0:
        t_end = 1e-3
    else:
        T = 2 * np.pi / osc.omega0
        t_end = duration_factor * T
    
    t_eval = np.linspace(0, t_end, points)
    sol = solve_ivp(
        harmonic_ode, [0, t_end], [osc.x0, osc.v0],
        args=(osc,), t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12
    )
    osc.t = sol.t
    osc.x = sol.y[0]
    osc.v = sol.y[1]
    return osc

def simulate_all(duration_factor=10, points=5000):
    """Simulate all hardware oscillators."""
    for name, osc in hardware_components.items():
        simulate_oscillator(osc, duration_factor, points)

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_all_time_domain():
    """Plot time domain responses of all oscillators."""
    simulate_all(duration_factor=10, points=5000)
    
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=False)
    fig.suptitle('Hardware Components as Harmonic Oscillators - Time Domain', 
                 fontsize=16, fontweight='bold')
    
    for idx, (name, osc) in enumerate(hardware_components.items()):
        ax = axes[idx]
        t_ms = osc.t * 1000  # Convert to milliseconds for display
        ax.plot(t_ms, osc.x, color=osc.color, linewidth=1.5)
        ax.set_ylabel(f'{name}', fontweight='bold', color=osc.color)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{osc.regime()}  |  ω₀={osc.omega0/(2*np.pi):.2e} Hz  |  Q={osc.quality_factor:.1f}')
        
        # Add annotation
        ax.text(0.02, 0.95, osc.desc[:80]+'...', transform=ax.transAxes,
                fontsize=7, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    return fig

def plot_phase_space():
    """Plot phase space (position vs velocity) for all oscillators."""
    simulate_all(duration_factor=20, points=10000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase Space Portraits (Position vs Velocity)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (name, osc) in enumerate(hardware_components.items()):
        ax = axes[idx]
        ax.plot(osc.x, osc.v, color=osc.color, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title(f'{name} (ζ={osc.zeta:.3f})', color=osc.color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add limit cycle or attractor indication
        if osc.zeta < 1:
            ax.text(0.95, 0.05, 'Spiral Attractor', transform=ax.transAxes,
                    ha='right', fontsize=8, fontstyle='italic')
        else:
            ax.text(0.95, 0.05, 'Node Attractor', transform=ax.transAxes,
                    ha='right', fontsize=8, fontstyle='italic')
    
    plt.tight_layout()
    return fig

def plot_frequency_response():
    """Plot FFT of each oscillator response."""
    simulate_all(duration_factor=50, points=20000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frequency Spectra (FFT)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (name, osc) in enumerate(hardware_components.items()):
        ax = axes[idx]
        dt = osc.t[1] - osc.t[0]
        n = len(osc.t)
        fft = np.abs(np.fft.fft(osc.x))[:n//2]
        freq = np.fft.fftfreq(n, dt)[:n//2]
        
        # Plot
        ax.semilogy(freq, fft, color=osc.color, linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{name}', color=osc.color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark natural frequency
        f0 = osc.omega0 / (2*np.pi)
        ax.axvline(f0, color='red', linestyle='--', alpha=0.5, label=f'ω₀={f0:.2e} Hz')
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    return fig

def plot_parameter_comparison():
    """Compare key parameters across all hardware components."""
    names = list(hardware_components.keys())
    colors = [osc.color for osc in hardware_components.values()]
    
    # Extract parameters
    zetas = [osc.zeta for osc in hardware_components.values()]
    q_factors = [min(osc.quality_factor, 1000) for osc in hardware_components.values()]  # Cap for visualization
    freqs = [osc.omega0/(2*np.pi) for osc in hardware_components.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Hardware Oscillator Parameter Comparison', fontsize=16, fontweight='bold')
    
    # Damping ratio
    ax = axes[0]
    bars = ax.bar(range(len(names)), zetas, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Damping Ratio ζ')
    ax.set_title('Damping Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label='Critical Damping')
    ax.legend()
    
    # Quality factor
    ax = axes[1]
    bars = ax.bar(range(len(names)), q_factors, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Quality Factor Q')
    ax.set_title('Q-Factor Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Natural frequency
    ax = axes[2]
    bars = ax.bar(range(len(names)), freqs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Natural Frequency (Hz)')
    ax.set_title('Frequency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def create_animated_oscillator():
    """Create an animated visualization of all oscillators."""
    simulate_all(duration_factor=10, points=500)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[3, 1])
    fig.suptitle('Live Harmonic Oscillator Watch', fontsize=16, fontweight='bold')
    
    # Time domain plots
    time_axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i, 0])
        time_axes.append(ax)
    
    # Summary panel
    summary_ax = fig.add_subplot(gs[:, 1])
    
    lines = []
    indicators = []
    
    for idx, (name, osc) in enumerate(hardware_components.items()):
        ax = time_axes[idx]
        ax.set_ylabel(name, color=osc.color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create initial empty lines
        line, = ax.plot([], [], color=osc.color, linewidth=2)
        lines.append(line)
        
        # Create indicator for current position
        ind, = ax.plot([], [], 'o', color=osc.color, markersize=8)
        indicators.append(ind)
        
        # Set axis limits
        ax.set_xlim(osc.t[0], osc.t[-1])
        if osc.x is not None:
            margin = (osc.x.max() - osc.x.min()) * 0.2 if osc.x.max() != osc.x.min() else 1
            ax.set_ylim(osc.x.min() - margin, osc.x.max() + margin)
        
        if idx == 5:
            ax.set_xlabel('Time (s)')
    
    # Summary panel
    summary_ax.set_xlim(0, 10)
    summary_ax.set_ylim(0, 6)
    summary_ax.axis('off')
    summary_ax.set_title('Component Status', fontweight='bold')
    
    # Create bars for summary
    bar_heights = [5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    bar_names = list(hardware_components.keys())
    bar_colors = [osc.color for osc in hardware_components.values()]
    
    bar_rects = []
    for i, (height, name, color) in enumerate(zip(bar_heights, bar_names, bar_colors)):
        rect = patches.Rectangle((1, height-0.3), 8, 0.6, 
                                  facecolor=color, alpha=0.3, edgecolor='black')
        summary_ax.add_patch(rect)
        summary_ax.text(0.5, height, name, ha='right', va='center', fontsize=9, fontweight='bold')
        bar_rects.append(rect)
    
    # Status texts
    status_texts = []
    for i, height in enumerate(bar_heights):
        text = summary_ax.text(9.2, height, '', va='center', fontsize=8)
        status_texts.append(text)
    
    def animate(frame):
        for idx, (name, osc) in enumerate(hardware_components.items()):
            # Update line
            lines[idx].set_data(osc.t[:frame+1], osc.x[:frame+1])
            
            # Update indicator
            if frame < len(osc.t):
                indicators[idx].set_data([osc.t[frame]], [osc.x[frame]])
            
            # Update bar width based on current amplitude
            if frame < len(osc.x):
                normalized_amplitude = abs(osc.x[frame] / max(abs(osc.x.max()), abs(osc.x.min()), 1))
                bar_rects[idx].set_width(2 + 6 * normalized_amplitude)
                
                # Update status text
                if osc.zeta < 1:
                    regime = 'UNDERDAMPED'
                elif np.isclose(osc.zeta, 1, rtol=1e-3):
                    regime = 'CRITICAL'
                else:
                    regime = 'OVERDAMPED'
                status_texts[idx].set_text(f'{regime} | x={osc.x[frame]:.3e}')
        
        return lines + indicators + bar_rects + status_texts
    
    ani = FuncAnimation(fig, animate, frames=len(list(hardware_components.values())[0].t),
                        interval=30, blit=True)
    
    plt.tight_layout()
    return ani

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Print component characteristics
    print("=" * 80)
    print("HARDWARE COMPONENTS AS HARMONIC OSCILLATORS")
    print("=" * 80)
    
    for name, osc in hardware_components.items():
        print(f"\n{name}:")
        print(f"  Regime: {osc.regime()}")
        print(f"  ω₀ = {osc.omega0/(2*np.pi):.3e} Hz")
        print(f"  ζ = {osc.zeta:.4f}")
        print(f"  Q = {osc.quality_factor:.2f}")
        print(f"  Description: {osc.desc}")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    # Time domain plot
    fig1 = plot_all_time_domain()
    plt.savefig('hardware_oscillators_time.png', dpi=150, bbox_inches='tight')
    print("✓ Time domain plots saved")
    
    # Phase space
    fig2 = plot_phase_space()
    plt.savefig('hardware_oscillators_phase.png', dpi=150, bbox_inches='tight')
    print("✓ Phase space plots saved")
    
    # Frequency response
    fig3 = plot_frequency_response()
    plt.savefig('hardware_oscillators_fft.png', dpi=150, bbox_inches='tight')
    print("✓ Frequency response plots saved")
    
    # Parameter comparison
    fig4 = plot_parameter_comparison()
    plt.savefig('hardware_oscillators_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Parameter comparison saved")
    
    # Show animation (uncomment to run)
    print("\nCreating animation...")
    ani = create_animated_oscillator()
    # To save: ani.save('hardware_oscillators.gif', writer='pillow', fps=30) 
    plt.show()