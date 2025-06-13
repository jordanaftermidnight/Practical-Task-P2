"""Advanced Signal Processing with NumPy and SciPy"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union
import warnings

class SignalProcessor:
    """
    Advanced signal processing operations using NumPy and SciPy.
    """
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def generate_complex_signal(self, duration: float = 1.0, 
                              frequencies: list = [10, 25, 50],
                              amplitudes: list = [1.0, 0.5, 0.3],
                              noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complex multi-frequency signal with noise.
        """
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Generate multi-frequency signal
        signal_clean = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal_clean += amp * np.sin(2 * np.pi * freq * t)
        
        # Add noise
        noise = noise_level * np.random.randn(len(t))
        signal_noisy = signal_clean + noise
        
        return t, signal_noisy
    
    def spectral_analysis(self, signal_data: np.ndarray, 
                         window: str = 'hann') -> dict:
        """
        Perform comprehensive spectral analysis.
        """
        # FFT-based power spectral density
        frequencies, psd = signal.welch(
            signal_data, 
            fs=self.sampling_rate, 
            window=window,
            nperseg=min(256, len(signal_data)//4)
        )
        
        # Spectrogram for time-frequency analysis
        f_spec, t_spec, Sxx = signal.spectrogram(
            signal_data, 
            fs=self.sampling_rate,
            window=window,
            nperseg=min(128, len(signal_data)//8)
        )
        
        # Peak detection in frequency domain
        peaks, properties = signal.find_peaks(
            psd, 
            height=np.max(psd) * 0.1,
            distance=10
        )
        
        # Spectral centroid and bandwidth
        spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
        spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        return {
            'frequencies': frequencies,
            'psd': psd,
            'spectrogram': {
                'frequencies': f_spec,
                'times': t_spec,
                'magnitude': 10 * np.log10(Sxx + 1e-12)
            },
            'peaks': {
                'frequencies': frequencies[peaks],
                'magnitudes': psd[peaks]
            },
            'spectral_features': {
                'centroid': spectral_centroid,
                'bandwidth': spectral_bandwidth,
                'rolloff': self._spectral_rolloff(frequencies, psd)
            }
        }
    
    def _spectral_rolloff(self, frequencies: np.ndarray, psd: np.ndarray, 
                         threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= threshold * cumsum_psd[-1])[0]
        return frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
    
    def advanced_filtering(self, signal_data: np.ndarray) -> dict:
        """
        Apply various advanced filtering techniques.
        """
        results = {}
        
        # Butterworth lowpass filter
        sos_butter = signal.butter(6, 100, 'low', fs=self.sampling_rate, output='sos')
        filtered_butter = signal.sosfilt(sos_butter, signal_data)
        results['butterworth_lowpass'] = filtered_butter
        
        # Chebyshev Type I bandpass filter
        sos_cheby = signal.cheby1(4, 1, [20, 200], 'band', fs=self.sampling_rate, output='sos')
        filtered_cheby = signal.sosfilt(sos_cheby, signal_data)
        results['chebyshev_bandpass'] = filtered_cheby
        
        # Elliptic highpass filter
        sos_ellip = signal.ellip(5, 1, 40, 50, 'high', fs=self.sampling_rate, output='sos')
        filtered_ellip = signal.sosfilt(sos_ellip, signal_data)
        results['elliptic_highpass'] = filtered_ellip
        
        # Savitzky-Golay filter for smoothing
        filtered_savgol = signal.savgol_filter(signal_data, 51, 3)
        results['savitzky_golay'] = filtered_savgol
        
        # Median filter for spike removal
        filtered_median = signal.medfilt(signal_data, kernel_size=5)
        results['median_filter'] = filtered_median
        
        # Wiener filter (deconvolution)
        # Create a simple impulse response for demonstration
        h = signal.windows.gaussian(21, std=3)
        h = h / np.sum(h)
        
        # Convolve signal with impulse response (simulate blurring)
        blurred = np.convolve(signal_data, h, mode='same')
        
        # Apply Wiener deconvolution
        filtered_wiener = signal.wiener(blurred, noise=0.01)
        results['wiener_deconvolution'] = filtered_wiener
        
        return results
    
    def wavelets_analysis(self, signal_data: np.ndarray) -> dict:
        """
        Perform wavelet-based analysis using continuous wavelet transform approximation.
        """
        # Morlet wavelet analysis using convolution
        def morlet_wavelet(t, f0=1.0, sigma=1.0):
            """Generate Morlet wavelet."""
            return (1 / np.sqrt(np.pi * sigma)) * np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-t**2 / sigma**2)
        
        # Create time vector for wavelets
        dt = 1 / self.sampling_rate
        t_wavelet = np.arange(-2, 2, dt)
        
        # Frequency range for analysis
        frequencies = np.logspace(0, 2, 50)  # 1 to 100 Hz
        
        # Compute continuous wavelet transform
        cwt_matrix = np.zeros((len(frequencies), len(signal_data)), dtype=complex)
        
        for i, freq in enumerate(frequencies):
            # Scale wavelet for current frequency
            scale = 1 / freq
            t_scaled = t_wavelet / scale
            wavelet = morlet_wavelet(t_scaled, f0=freq)
            
            # Convolve signal with wavelet
            cwt_row = np.convolve(signal_data, wavelet, mode='same')
            cwt_matrix[i, :] = cwt_row
        
        # Compute scalogram (time-frequency representation)
        scalogram = np.abs(cwt_matrix) ** 2
        
        # Ridge extraction (dominant frequency over time)
        ridge_indices = np.argmax(scalogram, axis=0)
        ridge_frequencies = frequencies[ridge_indices]
        
        return {
            'frequencies': frequencies,
            'cwt_matrix': cwt_matrix,
            'scalogram': scalogram,
            'ridge_frequencies': ridge_frequencies,
            'time_vector': np.arange(len(signal_data)) / self.sampling_rate
        }
    
    def nonlinear_analysis(self, signal_data: np.ndarray) -> dict:
        """
        Perform nonlinear signal analysis.
        """
        # Hilbert transform for instantaneous phase and amplitude
        analytic_signal = signal.hilbert(signal_data)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sampling_rate
        instantaneous_amplitude = np.abs(analytic_signal)
        
        # Empirical Mode Decomposition (simplified version)
        def simple_emd(signal_data, max_imfs=5):
            """Simplified EMD implementation."""
            imfs = []
            residue = signal_data.copy()
            
            for _ in range(max_imfs):
                # Find local maxima and minima
                maxima = signal.argrelextrema(residue, np.greater)[0]
                minima = signal.argrelextrema(residue, np.less)[0]
                
                if len(maxima) < 3 or len(minima) < 3:
                    break
                
                # Interpolate envelopes
                t = np.arange(len(residue))
                try:
                    upper_envelope = np.interp(t, maxima, residue[maxima])
                    lower_envelope = np.interp(t, minima, residue[minima])
                    mean_envelope = (upper_envelope + lower_envelope) / 2
                    
                    # Extract IMF
                    imf = residue - mean_envelope
                    imfs.append(imf)
                    residue = residue - imf
                except:
                    break
            
            imfs.append(residue)  # Add final residue
            return imfs
        
        imfs = simple_emd(signal_data)
        
        # Fractal dimension estimation (box counting method)
        def fractal_dimension(signal_data, max_box_size=None):
            """Estimate fractal dimension using box counting."""
            if max_box_size is None:
                max_box_size = len(signal_data) // 10
            
            # Normalize signal
            normalized = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
            
            box_sizes = np.logspace(0, np.log10(max_box_size), 20).astype(int)
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes needed to cover the signal
                n_boxes = int(np.ceil(len(normalized) / box_size))
                count = 0
                
                for i in range(n_boxes):
                    start_idx = i * box_size
                    end_idx = min((i + 1) * box_size, len(normalized))
                    box_range = np.max(normalized[start_idx:end_idx]) - np.min(normalized[start_idx:end_idx])
                    
                    if box_range > 0:
                        count += int(np.ceil(box_range * box_size))
                
                box_counts.append(count)
            
            # Estimate fractal dimension from slope
            log_boxes = np.log(box_sizes)
            log_counts = np.log(np.array(box_counts) + 1)
            
            # Linear regression
            coeffs = np.polyfit(log_boxes, log_counts, 1)
            fractal_dim = -coeffs[0]
            
            return fractal_dim, box_sizes, box_counts
        
        fractal_dim, box_sizes, box_counts = fractal_dimension(signal_data)
        
        return {
            'hilbert_transform': {
                'instantaneous_phase': instantaneous_phase,
                'instantaneous_frequency': instantaneous_frequency,
                'instantaneous_amplitude': instantaneous_amplitude
            },
            'emd': {
                'imfs': imfs,
                'n_imfs': len(imfs)
            },
            'fractal_analysis': {
                'dimension': fractal_dim,
                'box_sizes': box_sizes,
                'box_counts': box_counts
            }
        }