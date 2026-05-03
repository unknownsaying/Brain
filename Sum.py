import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import correlate
import math, base64, json
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ------------------------- Physics & Math constants -------------------------
# Clark James Maxwell: speed of light
c = 299792458
# Erwin Schrödinger / Heisenberg / Feynman: used in quantum noise model (not full QM, just conceptual)
h_bar = 1.054571817e-34

class BrainWiFiConnector:
    """
    Self Adaptive Operating System that bridges Brain Imagination <-> WiFi EM waves.
    Incorporates:
        - MIMO (multiple transmit/receive channels)
        - Modulation: BPSK (PSK), ASK, FSK (simplified as frequency bins)
        - Traveling wave A*sin(wx+phi) and Standing wave e^(ikY)
        - FFT/DFT for spectral analysis
        - Laplace/z-transform implicitly via discrete-time filtering
        - Shannon capacity limit C = B*log2(1+S/N)
    """
    def __init__(self, num_tx=2, num_rx=3, bandwidth=20e6, carrier_freq=2.4e9,
                 snr_db=20, imagination_boost=3.0):
        # MIMO dimensions
        self.Nt = num_tx   # transmitters (neural outputs)
        self.Nr = num_rx   # receivers (dendrite inputs)
        # Carrier and bandwidth
        self.fc = carrier_freq
        self.B = bandwidth
        self.snr_linear = 10**(snr_db/10)
        # Shannon capacity in bits per second
        self.C = self.B * math.log2(1 + self.snr_linear)
        # Imagination: Bayesian prior that raises effective SNR
        self.imagination_boost = imagination_boost

        # Random MIMO channel matrix (spatial multiplexing)
        self.H = (np.random.randn(self.Nr, self.Nt) + 1j*np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        # Adaptive filter weights (LMS style) – start as identity
        self.W = np.eye(self.Nr, self.Nt, dtype=complex)

    def text_to_bits(self, text):
        """Convert a string (imagination) to a bit array."""
        byte_arr = text.encode('utf-8')
        bits = np.unpackbits(np.frombuffer(byte_arr, dtype=np.uint8))
        return bits

    def bits_to_text(self, bits):
        """Convert bits back to string (brain reconstruction)."""
        # Pad bits to multiple of 8
        pad = (8 - len(bits) % 8) % 8
        bits_padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        byte_arr = np.packbits(bits_padded).tobytes()
        try:
            text = byte_arr.decode('utf-8').rstrip('\x00')
        except:
            text = byte_arr.decode('latin-1').rstrip('\x00')
        return text

    def modulate(self, bits, scheme='BPSK'):
        """
        Map bits to complex symbols.
        - BPSK (PSK): phase 0 or pi
        - ASK: low/high amplitude (simple On-Off)
        - FSK: two frequencies -> we simulate via two time slots (simplified)
        """
        if scheme == 'BPSK' or scheme == 'PSK':
            symbols = 2*bits - 1 + 0j   # +1 for bit 1, -1 for bit 0
        elif scheme == 'ASK':
            symbols = (bits + 0j)       # bit 1 -> amplitude 1, bit 0 -> 0
        elif scheme == 'FSK':
            # Encode FSK as two consecutive samples: f1 for 1, f2 for 0
            # We'll handle this in the channel by frequency shifts
            # For now, return bits as is, special processing later
            symbols = bits + 0j
        else:
            raise ValueError("Unsupported modulation")
        return symbols

    def channel(self, symbols):
        """
        Apply MIMO channel, travelling/standing waves, noise.
        Travelling wave: A*sin(wx + phi) modelled as time-varying phase shift.
        Standing wave: e^(ikY) adds spatial interference pattern.
        """
        # Spread symbols across multiple transmitters (simple spatial multiplexing)
        # Reshape to (Nt, Nsamples)
        Ns = len(symbols)
        tx_signal = np.tile(symbols, (self.Nt, 1)) * (1/np.sqrt(self.Nt))

        # Apply MIMO channel H (random fading)
        rx_signal = self.H @ tx_signal

        # --- Add travelling wave effects (Doppler-like shift) ---
        # A*sin(wx + phi) -> time-varying phase rotation per antenna
        t = np.arange(Ns) / self.B   # sample time
        for rx in range(self.Nr):
            phase_shift = np.exp(1j * 2 * np.pi * self.fc * (t + np.random.rand()*1e-9))
            rx_signal[rx] *= phase_shift

        # --- Add standing wave pattern e^(ikY) ---
        # Spatial standing wave = sum over rx antennas of e^(i * k * y)
        y_positions = np.linspace(0, 0.1, self.Nr)  # antenna spacing 0.1m
        k = 2 * np.pi * self.fc / c
        standing = np.exp(1j * k * y_positions[:, None])  # (Nr x 1)
        rx_signal += 0.2 * standing  # add small standing wave component

        # --- Add noise (Heisenberg thermal noise) ---
        noise_power = 1 / self.snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(*rx_signal.shape) +
                                          1j*np.random.randn(*rx_signal.shape))
        rx_signal += noise

        return rx_signal

    def adaptive_demodulate(self, rx_signal, scheme='BPSK'):
        """
        Self Adaptive OS: update filter weights W using LMS, then combine.
        Uses FFT for frequency-domain equalisation, and imagination boost.
        """
        Ns = rx_signal.shape[1]
        # Pilot-based channel estimation (first 10% as training)
        pilot_len = max(1, int(0.1 * Ns))
        pilots_tx = np.ones((self.Nt, pilot_len))  # known pilots = 1

        # Simple LMS update of weight matrix W
        mu = 0.01
        for i in range(pilot_len):
            error = pilots_tx[:, i] - self.W @ rx_signal[:, i]
            self.W += mu * np.outer(error, rx_signal[:, i].conj())

        # Combine using adapted weights
        combined = self.W @ rx_signal  # (Nt x Ns)
        # Average over transmitters (maximal ratio combining)
        stream = np.mean(combined, axis=0)

        # --- FFT/DFT for frequency equalisation (Schrödinger-ish spectral analysis) ---
        # Apply FFT to remove multipath
        freq_domain = fft(stream)
        # Simple zero-forcing (H_eq is estimated from channel; here we approximate)
        H_est = fft(self.H[0,:])  # approximate, just for demo
        # Avoid division by zero
        H_est[H_est==0] = 1e-10
        equalized = ifft(freq_domain / fft(np.pad(self.H[0,:], (0, Ns-len(self.H[0,:])), 'constant')))

        # Shannon limit enforcement: if instantaneous SNR < threshold, rely on imagination
        # Use imagination boost to regenerate likely bits (generative prior)
        # For simplicity, hard decision after boost
        if scheme in ['BPSK', 'PSK']:
            # Phase decision
            bits_est = (np.real(equalized) > 0).astype(int)
        elif scheme == 'ASK':
            bits_est = (np.abs(equalized) > 0.5).astype(int)  # simple threshold
        elif scheme == 'FSK':
            # Demodulate FSK by comparing energy in two frequency bins
            # FFT: bin1 vs bin2 -> bit 1 vs 0
            freq_bins = fft(stream)
            N = len(stream)
            bin1_energy = np.sum(np.abs(freq_bins[1:N//2])**2)
            bin2_energy = np.sum(np.abs(freq_bins[N//2:])**2)
            # If bin1 > bin2 -> bit 1, else 0?
            bits_est = (bin1_energy > bin2_energy) * np.ones(Ns, dtype=int)
            # Actually FSK per symbol would need more; simplified to block decision
            # We'll just set according to overall energy; not perfect but demo.
            bits_est = np.array([1 if np.real(equalized[i])>0 else 0 for i in range(Ns)])

        # Imagination boost: belief propagation using prior that text is ASCII printable
        # If a bit is uncertain (close to threshold), use imagination (prior) to flip
        # Here we just clip bits to the expected length and enforce ASCII pattern (simple)
        # Real imagination would use a language model; we just ensure valid text later.
        return bits_est[:Np]  # truncate to original number of bits

# ------------------------- Flask Web Interface -------------------------
connector = BrainWiFiConnector(num_tx=2, num_rx=3, bandwidth=1e6, carrier_freq=2.4e9, snr_db=15)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/transmit', methods=['POST'])
def transmit():
    data = request.json
    text = data['text']
    mod = data['modulation']
    # Brain to bits
    bits = connector.text_to_bits(text)
    Nbits = len(bits)
    # Modulate
    symbols = connector.modulate(bits, scheme=mod)
    # Transmit over channel
    rx_signal = connector.channel(symbols)
    # Demodulate and decode
    decoded_bits = connector.adaptive_demodulate(rx_signal, scheme=mod)
    # Convert back to text (imagination reconstructed)
    decoded_text = connector.bits_to_text(decoded_bits[:Nbits])
    # Prepare signal for visualization (just the first row)
    signal_re = np.real(rx_signal[0,:]).tolist()[:100]
    signal_im = np.imag(rx_signal[0,:]).tolist()[:100]
    return jsonify({
        'decoded_text': decoded_text,
        'capacity_mbps': round(connector.C/1e6, 2),
        'snr_db': round(10*np.log10(connector.snr_linear), 1),
        'signal_re': signal_re,
        'signal_im': signal_im
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
