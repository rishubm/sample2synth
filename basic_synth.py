import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
from dataclasses import dataclass
from typing import Literal

@dataclass
class SynthParams:
    """Parameters for our subtractive synthesizer"""
    # Oscillator
    osc_type: Literal['sine', 'saw', 'square', 'noise'] = 'saw'
    osc_mix: float = 1.0  # 0-1, could mix multiple oscillators later
    
    # Filter
    filter_cutoff: float = 1000.0  # Hz
    filter_resonance: float = 1.0  # Q factor
    filter_type: Literal['lowpass', 'highpass', 'bandpass'] = 'lowpass'
    
    # Envelope (ADSR)
    attack: float = 0.01   # seconds
    decay: float = 0.1     # seconds  
    sustain: float = 0.7   # level (0-1)
    release: float = 0.2   # seconds
    
    # General
    amplitude: float = 0.3  # overall volume

class SubtractiveSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def generate_oscillator(self, frequency, duration, osc_type='saw'):
        """Generate basic oscillator waveforms"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        if osc_type == 'sine':
            return np.sin(2 * np.pi * frequency * t)
        elif osc_type == 'saw':
            return signal.sawtooth(2 * np.pi * frequency * t)
        elif osc_type == 'square':
            return signal.square(2 * np.pi * frequency * t)
        elif osc_type == 'noise':
            return np.random.uniform(-1, 1, len(t))
        else:
            raise ValueError(f"Unknown oscillator type: {osc_type}")
    
    def apply_filter(self, audio_signal, cutoff, resonance, filter_type='lowpass'):
        """Apply digital filter to audio signal with resonance"""
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Clamp cutoff to valid range
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
        
        # Convert resonance (1.0-10.0) to Q factor for filter design
        # Higher resonance = higher Q = sharper peak at cutoff frequency
        Q = np.clip(resonance, 0.5, 20.0)  # Reasonable Q range
        
        if filter_type == 'lowpass':
            # Use elliptic or Chebyshev for resonance, or create custom IIR
            sos = signal.iirfilter(4, normalized_cutoff, btype='low', 
                                  ftype='butter', output='sos')
            # Apply resonance by creating a peak at cutoff frequency
            filtered = signal.sosfiltfilt(sos, audio_signal)
            
            # Add resonance boost by creating a bandpass peak at cutoff
            if resonance > 1.0:
                # Create narrow bandpass around cutoff for resonance peak
                bandwidth = normalized_cutoff / Q
                low_res = max(0.01, normalized_cutoff - bandwidth/2)
                high_res = min(0.99, normalized_cutoff + bandwidth/2)
                
                # Design resonance filter
                sos_res = signal.iirfilter(2, [low_res, high_res], 
                                         btype='bandpass', ftype='butter', output='sos')
                resonance_signal = signal.sosfiltfilt(sos_res, audio_signal)
                
                # Mix original filtered signal with resonance
                resonance_gain = (resonance - 1.0) * 0.3  # Scale resonance effect
                filtered = filtered + resonance_signal * resonance_gain
                
        elif filter_type == 'highpass':
            sos = signal.iirfilter(4, normalized_cutoff, btype='high', 
                                  ftype='butter', output='sos')
            filtered = signal.sosfiltfilt(sos, audio_signal)
            
            # Add resonance for highpass
            if resonance > 1.0:
                bandwidth = normalized_cutoff / Q
                low_res = max(0.01, normalized_cutoff - bandwidth/2)
                high_res = min(0.99, normalized_cutoff + bandwidth/2)
                
                sos_res = signal.iirfilter(2, [low_res, high_res], 
                                         btype='bandpass', ftype='butter', output='sos')
                resonance_signal = signal.sosfiltfilt(sos_res, audio_signal)
                
                resonance_gain = (resonance - 1.0) * 0.3
                filtered = filtered + resonance_signal * resonance_gain
                
        elif filter_type == 'bandpass':
            # For bandpass, use cutoff as center frequency
            bandwidth = normalized_cutoff / Q  # Narrower bandwidth = higher resonance
            low = max(0.01, normalized_cutoff - bandwidth/2)
            high = min(0.99, normalized_cutoff + bandwidth/2)
            
            sos = signal.iirfilter(4, [low, high], btype='bandpass', 
                                  ftype='butter', output='sos')
            filtered = signal.sosfiltfilt(sos, audio_signal)
        
        # Normalize to prevent clipping from resonance boost
        max_val = np.max(np.abs(filtered))
        if max_val > 1.0:
            filtered = filtered / max_val
        
        return filtered
    
    def generate_envelope(self, duration, attack, decay, sustain, release):
        """Generate ADSR envelope"""
        total_samples = int(self.sample_rate * duration)
        envelope = np.zeros(total_samples)
        
        # Convert times to samples
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Sustain samples is what's left
        sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)
        
        idx = 0
        
        # Attack phase
        if attack_samples > 0:
            envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples
        
        # Decay phase  
        if decay_samples > 0:
            envelope[idx:idx+decay_samples] = np.linspace(1, sustain, decay_samples)
            idx += decay_samples
            
        # Sustain phase
        if sustain_samples > 0:
            envelope[idx:idx+sustain_samples] = sustain
            idx += sustain_samples
            
        # Release phase
        if release_samples > 0 and idx < total_samples:
            remaining = total_samples - idx
            envelope[idx:] = np.linspace(sustain, 0, remaining)
        
        return envelope
    
    def synthesize(self, frequency, duration, params: SynthParams):
        """Generate a synthesized note with given parameters"""
        # Generate oscillator
        osc_signal = self.generate_oscillator(frequency, duration, params.osc_type)
        
        # Apply filter
        filtered_signal = self.apply_filter(
            osc_signal, 
            params.filter_cutoff, 
            params.filter_resonance, 
            params.filter_type
        )
        
        # Generate and apply envelope
        envelope = self.generate_envelope(
            duration, 
            params.attack, 
            params.decay, 
            params.sustain, 
            params.release
        )
        
        # Apply envelope and amplitude
        final_signal = filtered_signal * envelope * params.amplitude
        
        return final_signal

# Example usage and testing
if __name__ == "__main__":
    # Create synthesizer
    synth = SubtractiveSynth()
    
    # Test different sounds
    test_params = [
        SynthParams(osc_type='saw', filter_cutoff=2000, filter_resonance=1.0, attack=0.01, release=0.5),
        SynthParams(osc_type='saw', filter_cutoff=2000, filter_resonance=5.0, attack=0.01, release=0.5),
        SynthParams(osc_type='saw', filter_cutoff=2000, filter_resonance=20.0, attack=0.01, release=0.5),
        SynthParams(osc_type='square', filter_cutoff=800, filter_resonance=3.0, filter_type='lowpass', attack=0.05, release=0.3),

    ]
    
    # Generate some test notes
    frequency = 440  # A4
    duration = 2.0   # 2 seconds
    
    print("Generating test sounds...")
    for i, params in enumerate(test_params):
        print(f"Test {i+1}: {params.osc_type} oscillator, {params.filter_cutoff}Hz cutoff")
        audio = synth.synthesize(frequency, duration, params)
        
        # Play the sound (comment out if you don't want audio playback)
        sd.play(audio, synth.sample_rate)
        sd.wait()  # Wait until sound finishes
        
        # Plot waveform
        plt.figure(figsize=(12, 8))
        t = np.linspace(0, duration, len(audio))
        
        # Full waveform (shows envelope)
        plt.subplot(2, 2, 1)
        plt.plot(t, audio)
        plt.title(f'Full Waveform - Test {i+1} (Envelope Shape)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Zoomed waveform (shows actual oscillations)
        plt.subplot(2, 2, 2)
        zoom_start = 0.6  # Start at 0.6 seconds (during sustain phase)
        zoom_duration = 0.01  # Show 10ms (about 4-5 cycles at 440Hz)
        zoom_start_idx = int(zoom_start * synth.sample_rate)
        zoom_end_idx = int((zoom_start + zoom_duration) * synth.sample_rate)
        t_zoom = t[zoom_start_idx:zoom_end_idx]
        audio_zoom = audio[zoom_start_idx:zoom_end_idx]
        plt.plot(t_zoom, audio_zoom)
        plt.title(f'Zoomed Waveform - Test {i+1} (Actual Oscillations)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot spectrum
        plt.subplot(2, 2, 3)
        freqs, spectrum = signal.welch(audio, synth.sample_rate, nperseg=1024)
        plt.semilogy(freqs, spectrum)
        plt.title(f'Spectrum - Test {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.xlim(0, 5000)
        
        # Show envelope separately
        plt.subplot(2, 2, 4)
        envelope = synth.generate_envelope(duration, params.attack, params.decay, params.sustain, params.release)
        plt.plot(t, envelope)
        plt.title(f'ADSR Envelope - Test {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Envelope Level')
        
        plt.tight_layout()
        plt.show()
    
    print("Synthesizer test complete!")