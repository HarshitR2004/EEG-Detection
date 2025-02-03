import numpy as np
from scipy import stats
from scipy.signal import welch
import pywt
from sklearn.preprocessing import StandardScaler

class EEGFeatureExtractor:
    def __init__(self, signals=None, sampling_rate=256):
        """Initialize EEGFeatureExtractor with EEG signals and sampling rate."""
        self.signals = signals
        self.sampling_rate = sampling_rate

    def extract_time_domain_features(self):
       
        n_samples = self.signals.shape[0]
        features_list = []
        
        for sample in range(n_samples):
            sample_features = []
            for channel in range(self.signals.shape[1]):
                channel_signal = self.signals[sample, channel, :]
                
                # Basic statistics
                mean = np.mean(channel_signal)
                std = np.std(channel_signal)
                max_val = np.max(channel_signal)
                min_val = np.min(channel_signal)
                
                # Zero crossing rate
                zero_crossings = np.sum(np.abs(np.diff(np.signbit(channel_signal))))
                
                # Activity measures
                rms = np.sqrt(np.mean(channel_signal**2))
                mean_abs = np.mean(np.abs(channel_signal))
                
                # Shape statistics
                kurtosis = stats.kurtosis(channel_signal)
                skewness = stats.skew(channel_signal)
                
                # Hjorth parameters
                diff_first = np.diff(channel_signal)
                diff_second = np.diff(diff_first)
                mobility = np.std(diff_first) / np.std(channel_signal)
                complexity = np.std(diff_second) / np.std(diff_first)
                
                channel_features = [
                    mean, std, max_val, min_val, zero_crossings,
                    rms, mean_abs, kurtosis, skewness,
                    mobility, complexity
                ]
                sample_features.extend(channel_features)
            
            features_list.append(sample_features)
        
        return np.array(features_list)

    def extract_frequency_domain_features(self):
        """Extract frequency domain features from EEG signals.
        Args:
            signals: Shape (samples, channels, time_points)
            sampling_rate: Sampling frequency in Hz
        Returns:
            features: Shape (samples, n_features)
        """
        n_samples = self.signals.shape[0]
        features_list = []
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        for sample in range(n_samples):
            sample_features = []
            for channel in range(self.signals.shape[1]):
                channel_signal = self.signals[sample, channel, :]
                
                # Compute power spectral density
                freqs, psd = welch(channel_signal, fs=self.sampling_rate)
                
                # Calculate band powers
                band_powers = {}
                for band_name, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    band_powers[band_name] = np.mean(psd[mask])
                
                # Spectral edge frequency (95% of power)
                total_power = np.cumsum(psd)
                total_power = total_power / total_power[-1]
                spectral_edge = freqs[np.where(total_power >= 0.95)[0][0]]
                
                channel_features = [
                    band_powers['delta'],
                    band_powers['theta'],
                    band_powers['alpha'],
                    band_powers['beta'],
                    band_powers['gamma'],
                    spectral_edge
                ]
                sample_features.extend(channel_features)
            
            features_list.append(sample_features)
        
        return np.array(features_list)

    def extract_connectivity_features(self):
        """Extract connectivity features between EEG channels.
        Args:
            signals: Shape (samples, channels, time_points)
        Returns:
            features: Shape (samples, n_features)
        """
        n_samples = self.signals.shape[0]
        features_list = []
        
        for sample in range(n_samples):
            sample_features = []
            n_channels = self.signals.shape[1]
            
            # Cross-correlation and coherence
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    sig_i = self.signals[sample, i, :]
                    sig_j = self.signals[sample, j, :]
                    
                    # Cross-correlation
                    correlation = np.corrcoef(sig_i, sig_j)[0, 1]
                    
                    # Coherence (simplified)
                    coh = np.abs(np.correlate(sig_i, sig_j)).max()
                    
                    sample_features.extend([correlation, coh])
            
            features_list.append(sample_features)
        
        return np.array(features_list)

    def extract_wavelet_features(self):
        """Extract wavelet-based features from EEG signals.
        Args:
            signals: Shape (samples, channels, time_points)
        Returns:
            features: Shape (samples, n_features)
        """
        n_samples = self.signals.shape[0]
        features_list = []
        wavelet = 'db4'
        level = 4
        
        for sample in range(n_samples):
            sample_features = []
            for channel in range(self.signals.shape[1]):
                channel_signal = self.signals[sample, channel, :]
                
                # Decompose signal
                coeffs = pywt.wavedec(channel_signal, wavelet, level=level)
                
                # Extract features from each coefficient level
                for coeff in coeffs:
                    # Statistical features
                    feat_mean = np.mean(coeff)
                    feat_std = np.std(coeff)
                    feat_energy = np.sum(coeff**2)
                    
                    level_features = [feat_mean, feat_std, feat_energy]
                    sample_features.extend(level_features)
            
            features_list.append(sample_features)
        
        return np.array(features_list)

    def combine_features(self, normalize=True):
        """Combine all features and optionally normalize.
        Args:
            signals: Shape (samples, channels, time_points)
            normalize: Whether to normalize features
        Returns:
            features: Shape (samples, n_total_features)
        """
        # Extract all feature types
        time_features = self.extract_time_domain_features()
        freq_features = self.extract_frequency_domain_features()
        wavelet_features = self.extract_wavelet_features()
        conn_features = self.extract_connectivity_features()
        
        # Combine features
        all_features = np.concatenate([
            time_features,
            freq_features,
            wavelet_features,
            conn_features
        ], axis=1)
        
        if normalize:
            scaler = StandardScaler()
            all_features = scaler.fit_transform(all_features)
        
        return all_features