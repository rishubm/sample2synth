import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import soundfile as sf
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_path):
        """Load audio file and ensure correct sample rate"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_spectral_features(self, audio):
        """Extract frequency-domain features"""
        features = {}
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff (where 85% of energy is below)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate (relates to oscillator type)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Spectral contrast (harmonic vs percussive content)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        
        return features
    
    def extract_harmonic_features(self, audio):
        """Extract features related to harmonic content"""
        features = {}
        
        # Pitch detection
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=self.sample_rate)
        
        # Fundamental frequency estimation
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_median'] = np.median(f0_clean)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_median'] = 0
        
        # Harmonicity (how harmonic vs inharmonic)
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2) + 1e-10)
        features['harmonic_ratio'] = harmonic_ratio
        
        return features
    
    def extract_temporal_features(self, audio):
        """Extract time-domain features (envelope, attack, etc.)"""
        features = {}
        
        # Envelope extraction
        envelope = np.abs(librosa.stft(audio))
        envelope = np.mean(envelope, axis=0)  # Average across frequencies
        
        # Smooth the envelope
        envelope_smooth = signal.savgol_filter(envelope, window_length=min(21, len(envelope)//2*2+1), polyorder=3)
        
        # Attack time (time to reach peak)
        peak_idx = np.argmax(envelope_smooth)
        attack_time = peak_idx * 512 / self.sample_rate  # Convert to seconds
        features['attack_time'] = attack_time
        
        # Decay characteristics
        if peak_idx < len(envelope_smooth) - 1:
            # Find decay slope (after peak)
            decay_portion = envelope_smooth[peak_idx:]
            if len(decay_portion) > 10:
                # Fit exponential decay
                try:
                    t = np.arange(len(decay_portion))
                    log_env = np.log(decay_portion + 1e-10)
                    slope, _, _, _, _ = stats.linregress(t, log_env)
                    features['decay_slope'] = slope
                except:
                    features['decay_slope'] = 0
            else:
                features['decay_slope'] = 0
        else:
            features['decay_slope'] = 0
        
        # Sustain level (average of middle portion)
        mid_start = len(envelope_smooth) // 4
        mid_end = 3 * len(envelope_smooth) // 4
        if mid_end > mid_start:
            sustain_level = np.mean(envelope_smooth[mid_start:mid_end])
            features['sustain_level'] = sustain_level / (np.max(envelope_smooth) + 1e-10)
        else:
            features['sustain_level'] = 0
        
        # Release time (rough estimate from end)
        release_portion = envelope_smooth[-len(envelope_smooth)//4:]
        if len(release_portion) > 5:
            release_slope = np.mean(np.diff(release_portion))
            features['release_slope'] = release_slope
        else:
            features['release_slope'] = 0
        
        # Overall envelope shape descriptors
        features['envelope_peak_position'] = peak_idx / len(envelope_smooth)
        features['envelope_peak_value'] = np.max(envelope_smooth)
        features['envelope_final_value'] = envelope_smooth[-1]
        
        return features
    
    def extract_mfcc_features(self, audio, n_mfcc=13):
        """Extract MFCC features (timbral characteristics)"""
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        # Statistics of each MFCC coefficient
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        return features
    
    def extract_all_features(self, audio_path):
        """Extract all features from an audio file"""
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        all_features = {}
        
        try:
            # Extract different feature groups
            spectral_features = self.extract_spectral_features(audio)
            harmonic_features = self.extract_harmonic_features(audio)
            temporal_features = self.extract_temporal_features(audio)
            mfcc_features = self.extract_mfcc_features(audio)
            
            # Combine all features
            all_features.update(spectral_features)
            all_features.update(harmonic_features)
            all_features.update(temporal_features)
            all_features.update(mfcc_features)
            
            # Add basic audio statistics
            all_features['audio_length'] = len(audio) / self.sample_rate
            all_features['audio_max'] = np.max(np.abs(audio))
            all_features['audio_mean'] = np.mean(np.abs(audio))
            all_features['audio_std'] = np.std(audio)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
        
        return all_features
    
    def process_dataset(self, training_data_dir, output_file="features.csv"):
        """Process entire dataset and extract features"""
        training_path = Path(training_data_dir)
        audio_dir = training_path / "audio"
        metadata_path = training_path / "metadata.json"
        
        # Load metadata
        if not metadata_path.exists():
            print(f"Metadata file not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Processing {len(metadata)} audio files...")
        
        all_data = []
        failed_files = []
        
        for item in tqdm(metadata, desc="Extracting features"):
            audio_path = audio_dir / item['filename']
            
            if not audio_path.exists():
                print(f"Audio file not found: {audio_path}")
                failed_files.append(item['filename'])
                continue
            
            # Extract features
            features = self.extract_all_features(audio_path)
            
            if features is None:
                failed_files.append(item['filename'])
                continue
            
            # Combine with metadata and parameters
            row_data = {
                'sample_id': item['sample_id'],
                'filename': item['filename'],
                'frequency': item['frequency'],
                'duration': item['duration']
            }
            
            # Add synthesis parameters (our targets)
            for param_name, param_value in item['params'].items():
                row_data[f'target_{param_name}'] = param_value
            
            # Add extracted features
            row_data.update(features)
            
            all_data.append(row_data)
        
        if failed_files:
            print(f"Failed to process {len(failed_files)} files: {failed_files[:5]}...")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save features
        output_path = training_path / output_file
        df.to_csv(output_path, index=False)
        
        print(f"Features extracted and saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Feature columns: {df.shape[1] - 4}")  # Subtract metadata columns
        
        return df
    
    def analyze_features(self, features_df):
        """Analyze extracted features and their correlations with parameters"""
        print("\n=== Feature Analysis ===")
        
        # Separate features from targets
        target_cols = [col for col in features_df.columns if col.startswith('target_')]
        feature_cols = [col for col in features_df.columns if not col.startswith(('target_', 'sample_id', 'filename', 'frequency', 'duration'))]
        
        print(f"Target parameters: {len(target_cols)}")
        print(f"Extracted features: {len(feature_cols)}")
        
        # Check for missing values
        missing_features = features_df[feature_cols].isnull().sum()
        if missing_features.sum() > 0:
            print(f"\nFeatures with missing values:")
            print(missing_features[missing_features > 0])
        
        # Correlation analysis for numeric targets
        print(f"\n=== Correlations with Target Parameters ===")
        
        for target_col in target_cols:
            if features_df[target_col].dtype in ['float64', 'int64']:
                print(f"\nTop correlations with {target_col}:")
                correlations = features_df[feature_cols].corrwith(features_df[target_col]).abs().sort_values(ascending=False)
                print(correlations.head(5))
        
        return feature_cols, target_cols

# Main execution
if __name__ == "__main__":
    # Create feature extractor
    extractor = AudioFeatureExtractor()
    
    # Process the training dataset
    print("Starting feature extraction...")
    
    # Make sure to use the same directory as your training data
    training_data_dir = "synth_training_data"
    
    # Extract features
    features_df = extractor.process_dataset(training_data_dir)
    
    if features_df is not None:
        # Analyze features
        feature_cols, target_cols = extractor.analyze_features(features_df)
        
        print(f"\nFeature extraction complete!")
        print(f"Ready for machine learning training")
        print(f"Next step: Train models to predict synthesis parameters")
    else:
        print("Feature extraction failed!")