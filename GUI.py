import sys
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter, spectrogram
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QComboBox, QSlider, QGroupBox, QGridLayout,
                             QTabWidget, QSplitter, QProgressBar, QDial)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import math
import time

# ============================================================
# BRAIN-WIFI SIGNAL PROCESSING ENGINE
# ============================================================

class BrainWiFiEngine:
    """Core engine simulating Brain-WiFi interface with Maxwell Physics and Shannon Theory"""
    
    def __init__(self):
        self.fs = 1000  # Sampling frequency
        self.duration = 2.0  # seconds
        self.carrier_freq = 2.4e9  # WiFi carrier
        self.bandwidth = 20e6  # WiFi bandwidth
        self.snr_db = 20
        self.brain_state = 'calm'  # calm, focused, meditative
        self.modulation = 'BPSK'
        
        # Maxwell parameters
        self.c = 3e8
        self.permittivity = 8.854e-12
        self.permeability = 4 * np.pi * 1e-7
        
    def set_brain_state(self, state):
        """Set brain state affecting signal processing"""
        self.brain_state = state
        states = {'calm': {'alpha': 10, 'beta': 2, 'snr_boost': 1.0},
                  'focused': {'alpha': 5, 'beta': 20, 'snr_boost': 2.0},
                  'meditative': {'alpha': 8, 'beta': 1, 'snr_boost': 1.5}}
        return states.get(state, states['calm'])
    
    def generate_brain_waves(self):
        """Generate synthetic brain waves (EEG-like)"""
        t = np.linspace(0, self.duration, int(self.fs * self.duration))
        state = self.set_brain_state(self.brain_state)
        
        # Alpha waves (8-12 Hz)
        alpha = np.sin(2 * np.pi * state['alpha'] * t) * 10
        # Beta waves (12-30 Hz)
        beta = np.sin(2 * np.pi * state['beta'] * t) * 5
        # Theta waves (4-8 Hz)
        theta = np.sin(2 * np.pi * 6 * t) * 3
        
        brain_signal = alpha + beta + theta + np.random.normal(0, 1, len(t))
        return t, brain_signal
    
    def generate_wifi_signal(self, message_bits):
        """Generate WiFi signal with modulation"""
        t = np.linspace(0, self.duration, len(message_bits))
        
        if self.modulation == 'BPSK':
            # BPSK: Phase 0 or π
            carrier = np.cos(2 * np.pi * 100 * t)  # Scaled down for visualization
            signal = carrier * (2 * message_bits - 1)
        elif self.modulation == 'QPSK':
            # QPSK implementation
            carrier_i = np.cos(2 * np.pi * 100 * t)
            carrier_q = np.sin(2 * np.pi * 100 * t)
            signal = carrier_i * (2 * message_bits - 1)
        elif self.modulation == 'ASK':
            # Amplitude Shift Keying
            signal = np.cos(2 * np.pi * 100 * t) * message_bits
        else:  # FSK
            f1, f2 = 100, 200
            signal = np.where(message_bits == 1, 
                            np.sin(2 * np.pi * f1 * t),
                            np.sin(2 * np.pi * f2 * t))
        
        return t, signal
    
    def apply_channel_effects(self, signal):
        """Apply Maxwell-inspired channel effects (multipath, noise)"""
        # Multipath effect (standing waves)
        delay = int(0.01 * self.fs)
        multipath = np.zeros_like(signal)
        if len(signal) > delay:
            multipath[delay:] = 0.5 * signal[:-delay]
        
        # Noise based on Heisenberg uncertainty principle
        noise_power = 10 ** (-self.snr_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        
        return signal + multipath + noise
    
    def shannon_capacity(self):
        """Calculate Shannon channel capacity"""
        return self.bandwidth * math.log2(1 + 10 ** (self.snr_db / 10))
    
    def maxwell_wave_propagation(self, distance):
        """Calculate EM wave propagation using Maxwell's equations"""
        # Free space path loss (Friis equation)
        wavelength = self.c / self.carrier_freq
        path_loss = (4 * np.pi * distance / wavelength) ** 2
        
        # Impedance of free space (Maxwell)
        z0 = np.sqrt(self.permeability / self.permittivity)
        
        # Poynting vector magnitude
        e_field = np.sqrt(2 * z0 * 1e-3 / (4 * np.pi * distance ** 2))
        
        return path_loss, e_field

# ============================================================
# BRAIN-WIFI GUI APPLICATION
# ============================================================

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

class BrainWiFiGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = BrainWiFiEngine()
        self.init_ui()
        self.running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        
    def init_ui(self):
        """Initialize the GUI interface"""
        self.setWindowTitle("🧠 Brain ⇄ WiFi Receiver - Neural Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                background-color: #16213e;
                color: #0f3460;
                border: 2px solid #0f3460;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0f3460;
                color: #e94560;
            }
            QPushButton:pressed {
                background-color: #e94560;
                color: white;
            }
            QTextEdit {
                background-color: #16213e;
                color: #e0e0e0;
                border: 2px solid #0f3460;
                border-radius: 5px;
                font-size: 14px;
            }
            QComboBox {
                background-color: #16213e;
                color: #e0e0e0;
                border: 2px solid #0f3460;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #16213e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QGroupBox {
                color: #e94560;
                border: 2px solid #0f3460;
                border-radius: 10px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QProgressBar {
                border: 2px solid #0f3460;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #16213e;
            }
            QProgressBar::chunk {
                background-color: #e94560;
                border-radius: 3px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🧠 BRAIN ⇄ WIFI RECEIVER INTERFACE ⚡")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #e94560; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #0f3460;
                border-radius: 10px;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #16213e;
                color: #e0e0e0;
                padding: 10px;
                margin-right: 5px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #e94560;
                color: white;
            }
        """)
        
        # Tab 1: Main Interface
        main_tab = QWidget()
        tabs.addTab(main_tab, "🔮 Neural Interface")
        self.setup_main_tab(main_tab)
        
        # Tab 2: Physics & Mathematics
        physics_tab = QWidget()
        tabs.addTab(physics_tab, "⚛️ Physics & Math")
        self.setup_physics_tab(physics_tab)
        
        # Tab 3: Signal Analysis
        signal_tab = QWidget()
        tabs.addTab(signal_tab, "📊 Signal Analysis")
        self.setup_signal_tab(signal_tab)
        
        # Tab 4: System Status
        status_tab = QWidget()
        tabs.addTab(status_tab, "💻 System Status")
        self.setup_status_tab(status_tab)
        
        main_layout.addWidget(tabs)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ START BRAIN-WIFI LINK")
        self.start_btn.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.start_btn)
        
        self.transmit_btn = QPushButton("📡 TRANSMIT THOUGHTS")
        self.transmit_btn.clicked.connect(self.transmit_signal)
        self.transmit_btn.setEnabled(False)
        control_layout.addWidget(self.transmit_btn)
        
        self.decode_btn = QPushButton("🧠 DECODE RECEIVED")
        self.decode_btn.clicked.connect(self.decode_signal)
        self.decode_btn.setEnabled(False)
        control_layout.addWidget(self.decode_btn)
        
        self.reset_btn = QPushButton("🔄 RESET SYSTEM")
        self.reset_btn.clicked.connect(self.reset_system)
        control_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(control_layout)
        
        # Status bar
        self.status_label = QLabel("Ready - Awaiting Neural Connection")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 12px; padding: 5px;")
        main_layout.addWidget(self.status_label)
    
    def setup_main_tab(self, tab):
        """Setup the main neural interface tab"""
        layout = QHBoxLayout(tab)
        
        # Left panel - Brain input
        left_panel = QVBoxLayout()
        
        # Brain state selection
        brain_group = QGroupBox("🧠 Brain State")
        brain_layout = QVBoxLayout()
        
        self.brain_combo = QComboBox()
        self.brain_combo.addItems(['calm', 'focused', 'meditative'])
        brain_layout.addWidget(QLabel("Mental State:"))
        brain_layout.addWidget(self.brain_combo)
        
        # Brain wave visualization
        self.brain_canvas = MplCanvas(self, width=4, height=3)
        brain_layout.addWidget(self.brain_canvas)
        
        brain_group.setLayout(brain_layout)
        left_panel.addWidget(brain_group)
        
        # Thought input
        thought_group = QGroupBox("💭 Thought Transmission")
        thought_layout = QVBoxLayout()
        
        self.thought_input = QTextEdit()
        self.thought_input.setPlaceholderText("Enter your thoughts to transmit via WiFi...")
        self.thought_input.setMaximumHeight(100)
        thought_layout.addWidget(self.thought_input)
        
        # Modulation selection
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(['BPSK', 'QPSK', 'ASK', 'FSK'])
        thought_layout.addWidget(QLabel("Modulation Scheme:"))
        thought_layout.addWidget(self.mod_combo)
        
        # SNR control
        thought_layout.addWidget(QLabel("Signal-to-Noise Ratio (dB):"))
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setMinimum(0)
        self.snr_slider.setMaximum(50)
        self.snr_slider.setValue(20)
        self.snr_label = QLabel("20 dB")
        self.snr_slider.valueChanged.connect(
            lambda v: self.snr_label.setText(f"{v} dB"))
        thought_layout.addWidget(self.snr_slider)
        thought_layout.addWidget(self.snr_label)
        
        thought_group.setLayout(thought_layout)
        left_panel.addWidget(thought_group)
        
        layout.addLayout(left_panel)
        
        # Right panel - Signal visualization
        right_panel = QVBoxLayout()
        
        # WiFi signal display
        wifi_group = QGroupBox("📡 WiFi Signal")
        wifi_layout = QVBoxLayout()
        self.wifi_canvas = MplCanvas(self, width=5, height=3)
        wifi_layout.addWidget(self.wifi_canvas)
        wifi_group.setLayout(wifi_layout)
        right_panel.addWidget(wifi_group)
        
        # Combined brain-WiFi display
        combined_group = QGroupBox("🔄 Brain-WiFi Integration")
        combined_layout = QVBoxLayout()
        self.combined_canvas = MplCanvas(self, width=5, height=3)
        combined_layout.addWidget(self.combined_canvas)
        combined_group.setLayout(combined_layout)
        right_panel.addWidget(combined_group)
        
        layout.addLayout(right_panel)
    
    def setup_physics_tab(self, tab):
        """Setup the physics and mathematics tab"""
        layout = QGridLayout(tab)
        
        # Maxwell's Equations display
        maxwell_group = QGroupBox("⚡ Maxwell's Electromagnetic Equations")
        maxwell_layout = QVBoxLayout()
        
        maxwell_text = QLabel("""
        ∇·E = ρ/ε₀          Gauss's Law
        ∇·B = 0             Gauss's Law for Magnetism
        ∇×E = -∂B/∂t        Faraday's Law
        ∇×B = μ₀J + μ₀ε₀∂E/∂t  Ampère-Maxwell Law
        
        Wave Equation: ∇²E - μ₀ε₀∂²E/∂t² = 0
        """)
        maxwell_text.setStyleSheet("font-family: monospace; font-size: 12px; color: #00ff00;")
        maxwell_layout.addWidget(maxwell_text)
        
        # EM wave propagation canvas
        self.em_canvas = MplCanvas(self, width=5, height=3)
        maxwell_layout.addWidget(self.em_canvas)
        
        maxwell_group.setLayout(maxwell_layout)
        layout.addWidget(maxwell_group, 0, 0)
        
        # Shannon Information Theory
        shannon_group = QGroupBox("📊 Shannon-Hartley Theorem")
        shannon_layout = QVBoxLayout()
        
        self.shannon_label = QLabel()
        shannon_layout.addWidget(self.shannon_label)
        
        # Capacity visualization
        self.shannon_canvas = MplCanvas(self, width=5, height=3)
        shannon_layout.addWidget(self.shannon_canvas)
        
        shannon_group.setLayout(shannon_layout)
        layout.addWidget(shannon_group, 0, 1)
        
        # Oscillator components
        osc_group = QGroupBox("🔄 Harmonic Oscillator Models")
        osc_layout = QVBoxLayout()
        self.osc_canvas = MplCanvas(self, width=5, height=3)
        osc_layout.addWidget(self.osc_canvas)
        osc_group.setLayout(osc_layout)
        layout.addWidget(osc_group, 1, 0)
        
        # Quantum effects
        quantum_group = QGroupBox("🔬 Quantum Considerations")
        quantum_layout = QVBoxLayout()
        
        quantum_text = QLabel("""
        Δx·Δp ≥ ℏ/2        Heisenberg Uncertainty
        iℏ∂Ψ/∂t = ĤΨ      Schrödinger Equation
        C = B·log₂(1+S/N)  Shannon Channel Capacity
        """)
        quantum_text.setStyleSheet("font-family: monospace; font-size: 12px; color: #00ffff;")
        quantum_layout.addWidget(quantum_text)
        
        quantum_group.setLayout(quantum_layout)
        layout.addWidget(quantum_group, 1, 1)
    
    def setup_signal_tab(self, tab):
        """Setup signal analysis tab"""
        layout = QGridLayout(tab)
        
        # FFT Spectrum
        fft_group = QGroupBox("📈 FFT Spectrum Analysis")
        fft_layout = QVBoxLayout()
        self.fft_canvas = MplCanvas(self, width=5, height=3)
        fft_layout.addWidget(self.fft_canvas)
        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group, 0, 0)
        
        # Spectrogram
        spectro_group = QGroupBox("🌊 Spectrogram")
        spectro_layout = QVBoxLayout()
        self.spectro_canvas = MplCanvas(self, width=5, height=3)
        spectro_layout.addWidget(self.spectro_canvas)
        spectro_group.setLayout(spectro_layout)
        layout.addWidget(spectro_group, 0, 1)
        
        # Phase space
        phase_group = QGroupBox("🌀 Phase Space")
        phase_layout = QVBoxLayout()
        self.phase_canvas = MplCanvas(self, width=5, height=3)
        phase_layout.addWidget(self.phase_canvas)
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group, 1, 0)
        
        # Constellation diagram
        const_group = QGroupBox("⭐ Constellation Diagram")
        const_layout = QVBoxLayout()
        self.const_canvas = MplCanvas(self, width=5, height=3)
        const_layout.addWidget(self.const_canvas)
        const_group.setLayout(const_layout)
        layout.addWidget(const_group, 1, 1)
    
    def setup_status_tab(self, tab):
        """Setup system status tab"""
        layout = QVBoxLayout(tab)
        
        # Connection status
        conn_group = QGroupBox("🔗 Connection Status")
        conn_layout = QGridLayout()
        
        self.conn_progress = QProgressBar()
        self.conn_progress.setMaximum(100)
        conn_layout.addWidget(QLabel("Link Quality:"), 0, 0)
        conn_layout.addWidget(self.conn_progress, 0, 1)
        
        self.signal_strength = QLabel("Signal Strength: -45 dBm")
        conn_layout.addWidget(self.signal_strength, 1, 0, 1, 2)
        
        self.bitrate_label = QLabel("Bitrate: 54 Mbps")
        conn_layout.addWidget(self.bitrate_label, 2, 0, 1, 2)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # System metrics
        metrics_group = QGroupBox("📊 System Metrics")
        metrics_layout = QGridLayout()
        
        metrics = [
            ("CPU Load:", "23%"),
            ("GPU Temp:", "56°C"),
            ("RAM Usage:", "4.2 GB / 16 GB"),
            ("Disk I/O:", "125 MB/s"),
            ("RISC-V Clock:", "1.8 GHz"),
            ("RLC Resonance:", "1.59 kHz")
        ]
        
        for i, (label, value) in enumerate(metrics):
            metrics_layout.addWidget(QLabel(label), i, 0)
            metrics_layout.addWidget(QLabel(value), i, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # System log
        log_group = QGroupBox("📝 System Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
    
    def toggle_connection(self):
        """Start/stop the brain-WiFi connection"""
        if not self.running:
            self.running = True
            self.start_btn.setText("⏸️ STOP BRAIN-WIFI LINK")
            self.start_btn.setStyleSheet("background-color: #e94560; color: white;")
            self.transmit_btn.setEnabled(True)
            self.decode_btn.setEnabled(True)
            self.timer.start(100)  # Update every 100ms
            self.status_label.setText("🟢 Brain-WiFi Connection Active")
            self.log("Brain-WiFi neural interface initialized")
            self.log("Maxwell EM field established")
            self.log("Shannon channel capacity: {:.2f} Mbps".format(
                self.engine.shannon_capacity() / 1e6))
        else:
            self.running = False
            self.start_btn.setText("▶ START BRAIN-WIFI LINK")
            self.start_btn.setStyleSheet("")
            self.transmit_btn.setEnabled(False)
            self.decode_btn.setEnabled(False)
            self.timer.stop()
            self.status_label.setText("⏸️ Connection Stopped")
            self.log("Brain-WiFi connection terminated")
    
    def transmit_signal(self):
        """Transmit thoughts via WiFi"""
        thought = self.thought_input.toPlainText()
        if thought:
            # Convert to binary
            bits = np.unpackbits(np.frombuffer(thought.encode(), dtype=np.uint8))
            
            # Get current settings
            self.engine.brain_state = self.brain_combo.currentText()
            self.engine.modulation = self.mod_combo.currentText()
            self.engine.snr_db = self.snr_slider.value()
            
            # Generate signals
            t, brain_wave = self.engine.generate_brain_waves()
            t_wifi, wifi_signal = self.engine.generate_wifi_signal(
                np.resize(bits, len(t)) if len(bits) < len(t) else bits[:len(t)])
            
            # Apply channel effects
            received = self.engine.apply_channel_effects(wifi_signal)
            
            self.log(f"Transmitted thought: '{thought[:30]}...'")
            self.log(f"Modulation: {self.engine.modulation}")
            self.log(f"Brain state: {self.engine.brain_state}")
            self.log(f"SNR: {self.engine.snr_db} dB")
            
            # Store for visualization
            self.current_brain = (t, brain_wave)
            self.current_wifi = (t_wifi, wifi_signal)
            self.current_received = (t_wifi, received)
    
    def decode_signal(self):
        """Decode received WiFi signal"""
        if hasattr(self, 'current_received'):
            # Simple BPSK decoding
            received = self.current_received[1]
            decoded_bits = (received > 0).astype(int)
            
            # Convert to text
            byte_arr = np.packbits(decoded_bits[:len(decoded_bits)//8*8])
            try:
                decoded_text = byte_arr.tobytes().decode('utf-8', errors='ignore')
                self.log(f"Decoded message: {decoded_text[:50]}")
            except:
                self.log("Decoding failed - signal too noisy")
            
            # Update connection quality
            quality = np.random.randint(60, 95)
            self.conn_progress.setValue(quality)
            self.signal_strength.setText(f"Signal Strength: -{np.random.randint(30, 70)} dBm")
    
    def reset_system(self):
        """Reset the entire system"""
        self.running = False
        self.timer.stop()
        self.start_btn.setText("▶ START BRAIN-WIFI LINK")
        self.start_btn.setStyleSheet("")
        self.transmit_btn.setEnabled(False)
        self.decode_btn.setEnabled(False)
        self.thought_input.clear()
        self.log_text.clear()
        self.conn_progress.setValue(0)
        self.status_label.setText("System Reset Complete")
        self.log("System reset - all connections terminated")
    
    def update_visualization(self):
        """Update all visualizations"""
        if not self.running:
            return
        
        # Update brain wave plot
        if hasattr(self, 'current_brain'):
            t, brain = self.current_brain
            self.brain_canvas.fig.clear()
            ax = self.brain_canvas.fig.add_subplot(111)
            ax.plot(t[:500], brain[:500], 'cyan', linewidth=1)
            ax.set_title('Brain Waves', color='white')
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            self.brain_canvas.draw()
        
        # Update WiFi signal plot
        if hasattr(self, 'current_wifi'):
            t, wifi = self.current_wifi
            self.wifi_canvas.fig.clear()
            ax = self.wifi_canvas.fig.add_subplot(111)
            ax.plot(t[:500], wifi[:500], '#e94560', linewidth=1)
            ax.set_title('WiFi Signal', color='white')
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            self.wifi_canvas.draw()
        
        # Update combined plot
        if hasattr(self, 'current_brain') and hasattr(self, 'current_wifi'):
            self.combined_canvas.fig.clear()
            ax = self.combined_canvas.fig.add_subplot(111)
            t_b, brain = self.current_brain
            t_w, wifi = self.current_wifi
            ax.plot(t_b[:500], brain[:500] / brain.max(), 'cyan', alpha=0.5, label='Brain')
            ax.plot(t_w[:500], wifi[:500] / wifi.max(), '#e94560', alpha=0.5, label='WiFi')
            ax.set_title('Brain-WiFi Integration', color='white')
            ax.legend()
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            self.combined_canvas.draw()
        
        # Update physics tab
        self.update_physics_display()
        
        # Update Shannon capacity
        capacity = self.engine.shannon_capacity() / 1e6
        self.shannon_label.setText(f"Channel Capacity: {capacity:.2f} Mbps")
        self.bitrate_label.setText(f"Current Bitrate: {capacity:.1f} Mbps")
        
        # Update connection quality
        if self.conn_progress.value() < 100:
            self.conn_progress.setValue(min(100, self.conn_progress.value() + 1))
    
    def update_physics_display(self):
        """Update physics visualizations"""
        # EM wave propagation
        self.em_canvas.fig.clear()
        ax = self.em_canvas.fig.add_subplot(111)
        x = np.linspace(0, 4*np.pi, 100)
        e_field = np.sin(x)
        b_field = np.cos(x)
        ax.plot(x, e_field, '#e94560', label='E-field', linewidth=2)
        ax.plot(x, b_field, '#00ff00', label='B-field', linewidth=2)
        ax.set_title('Electromagnetic Wave Propagation', color='white')
        ax.legend()
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        self.em_canvas.draw()
        
        # Shannon capacity curve
        self.shannon_canvas.fig.clear()
        ax = self.shannon_canvas.fig.add_subplot(111)
        snr_range = np.linspace(0, 30, 100)
        capacity = 20e6 * np.log2(1 + 10 ** (snr_range / 10))
        ax.plot(snr_range, capacity / 1e6, '#00ffff', linewidth=2)
        ax.set_xlabel('SNR (dB)', color='white')
        ax.set_ylabel('Capacity (Mbps)', color='white')
        ax.set_title('Shannon Channel Capacity', color='white')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        self.shannon_canvas.draw()
        
        # Oscillator visualization
        self.osc_canvas.fig.clear()
        ax = self.osc_canvas.fig.add_subplot(111)
        t = np.linspace(0, 10, 1000)
        # Multiple oscillators with different damping
        for zeta, label, color in [(0.1, 'CPU', '#ff6b6b'), 
                                    (0.5, 'GPU', '#ffd93d'),
                                    (1.0, 'RAM', '#6bff6b'),
                                    (2.0, 'RLC', '#ff6bff')]:
            omega = 2 * np.pi * 1
            x = np.exp(-zeta * omega * t) * np.cos(omega * np.sqrt(1 - min(zeta**2, 0.99)) * t)
            ax.plot(t, x + list([0.1, 0.5, 1.0, 2.0]).index(zeta)*3, 
                   color=color, label=f'{label} (ζ={zeta})', alpha=0.7)
        ax.set_title('Hardware Harmonic Oscillators', color='white')
        ax.legend()
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        self.osc_canvas.draw()
    
    def log(self, message):
        """Add message to system log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    app.setPalette(palette)
    
    window = BrainWiFiGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    
    main()