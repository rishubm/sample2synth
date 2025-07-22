import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from audio_feature_extractor import AudioFeatureExtractor
from model import SynthParameterPredictor
from basic_synth import SubtractiveSynth, SynthParams
import soundfile as sf
import matplotlib.pyplot as plt

class SynthInference:
    def __init__(self, model_dir="trained_models"):
        """Initialize inference system with trained models"""
        self.feature_extractor = AudioFeatureExtractor()
        self.predictor = SynthParameterPredictor(model_dir=model_dir)
        self.synth = SubtractiveSynth()
        
        # Load trained models
        try:
            self.predictor.load_models()
            print(f"✅ Loaded models from {model_dir}")
            print(f"   Available parameters: {list(self.predictor.models.keys())}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("   Make sure you've trained the models first by running model.py")
            raise
    
    def analyze_audio_file(self, audio_path):
        """Analyze an audio file and predict synth parameters"""
        print(f"\n🎵 Analyzing: {audio_path}")
        
        # Extract features from audio
        print("   Extracting audio features...")
        features = self.feature_extractor.extract_all_features(audio_path)
        
        if features is None:
            raise ValueError(f"Failed to extract features from {audio_path}")
        
        print(f"   Extracted {len(features)} features")
        
        # Predict synthesis parameters
        print("   Predicting synthesis parameters...")
        predictions = self.predictor.predict_parameters(features)
        
        return features, predictions
    
    def synthesize_from_predictions(self, predictions, features=None, frequency=None, duration=2.0):
        """Create audio using predicted parameters"""
        print("   Generating audio from predictions...")
        
        # Use detected frequency from features if not explicitly provided
        if frequency is None and features is not None:
            detected_freq = features.get('f0_mean', 440.0)
            # Only use detected frequency if it's reasonable (not 0 or too extreme)
            if 80 <= detected_freq <= 2000:  # Reasonable musical range
                frequency = detected_freq
                print(f"   Using detected frequency: {frequency:.1f} Hz")
            else:
                frequency = 440.0
                print(f"   Detected frequency {detected_freq:.1f} Hz out of range, using 440 Hz")
        elif frequency is None:
            frequency = 440.0
            print("   Using default frequency: 440 Hz")
        else:
            print(f"   Using provided frequency: {frequency:.1f} Hz")
        
        # Convert predictions to SynthParams
        params = SynthParams(
            osc_type=predictions.get('osc_type', 'saw'),
            filter_cutoff=predictions.get('filter_cutoff', 1000.0),
            filter_resonance=predictions.get('filter_resonance', 1.0),
            filter_type=predictions.get('filter_type', 'lowpass'),
            attack=predictions.get('attack', 0.01),
            decay=predictions.get('decay', 0.1),
            sustain=predictions.get('sustain', 0.7),
            release=predictions.get('release', 0.2),
            amplitude=predictions.get('amplitude', 0.3)
        )
        
        # Generate audio
        audio = self.synth.synthesize(frequency, duration, params)
        
        return audio, params, frequency
    
    def compare_with_original(self, original_path, synthesized_audio, output_dir="comparison_output"):
        """Compare original and synthesized audio"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load original audio
        original_audio = self.feature_extractor.load_audio(original_path)
        if original_audio is None:
            return
        
        # Save synthesized audio
        synth_path = output_path / f"synthesized_{Path(original_path).stem}.wav"
        sf.write(synth_path, synthesized_audio, self.synth.sample_rate)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain comparison
        t_orig = np.linspace(0, len(original_audio)/self.feature_extractor.sample_rate, len(original_audio))
        t_synth = np.linspace(0, len(synthesized_audio)/self.synth.sample_rate, len(synthesized_audio))
        
        axes[0, 0].plot(t_orig, original_audio, alpha=0.7, label='Original')
        axes[0, 0].set_title('Original Audio Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        axes[0, 1].plot(t_synth, synthesized_audio, alpha=0.7, label='Synthesized', color='orange')
        axes[0, 1].set_title('Synthesized Audio Waveform')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        
        # Frequency domain comparison
        from scipy import signal
        
        freqs_orig, spectrum_orig = signal.welch(original_audio, self.feature_extractor.sample_rate, nperseg=1024)
        freqs_synth, spectrum_synth = signal.welch(synthesized_audio, self.synth.sample_rate, nperseg=1024)
        
        axes[1, 0].semilogy(freqs_orig, spectrum_orig, alpha=0.7)
        axes[1, 0].set_title('Original Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_xlim(0, 5000)
        
        axes[1, 1].semilogy(freqs_synth, spectrum_synth, alpha=0.7, color='orange')
        axes[1, 1].set_title('Synthesized Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_xlim(0, 5000)
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_path = output_path / f"comparison_{Path(original_path).stem}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Comparison plot saved: {plot_path}")
        
        plt.show()
        
        return synth_path, plot_path
    
    def run_full_analysis(self, audio_path, output_dir="analysis_output", frequency=None, duration=2.0):
        """Run complete analysis pipeline"""
        print("="*60)
        print("🎹 SAMPLE TO SYNTHESIS ANALYSIS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Step 1: Analyze audio file
            features, predictions = self.analyze_audio_file(audio_path)
            
            # Step 2: Generate synthesized audio (using detected frequency if not provided)
            synthesized_audio, synth_params, used_frequency = self.synthesize_from_predictions(
                predictions, features, frequency, duration
            )
            
            # Step 3: Save results
            results = {
                'input_file': str(audio_path),
                'detected_frequency': features.get('f0_mean', 0),
                'synthesis_frequency': used_frequency,
                'predicted_parameters': predictions,
                'extracted_features': features,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save results as JSON
            results_path = output_path / f"analysis_{Path(audio_path).stem}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Step 4: Compare with original
            synth_path, plot_path = self.compare_with_original(
                audio_path, synthesized_audio, output_dir
            )
            
            # Step 5: Print results
            print(f"\n🎵 FREQUENCY ANALYSIS:")
            print("-" * 40)
            print(f"   Detected frequency: {features.get('f0_mean', 0):.1f} Hz")
            print(f"   Synthesis frequency: {used_frequency:.1f} Hz")
            
            print("\n📋 PREDICTED PARAMETERS:")
            print("-" * 40)
            for param, value in predictions.items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.4f}")
                else:
                    print(f"   {param}: {value}")
            
            print(f"\n💾 FILES SAVED:")
            print(f"   📄 Analysis results: {results_path}")
            print(f"   🎵 Synthesized audio: {synth_path}")
            print(f"   📊 Comparison plot: {plot_path}")
            
            return results, synth_path
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise

def test_with_training_samples(inference_system, training_data_dir="synth_training_data", num_samples=5):
    """Test the inference system with samples from the training dataset"""
    print("\n" + "="*60)
    print("🧪 TESTING WITH TRAINING SAMPLES")
    print("="*60)
    
    # Load training metadata
    metadata_path = Path(training_data_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ Training metadata not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Test with a few random samples
    import random
    test_samples = random.sample(metadata, min(num_samples, len(metadata)))
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Test {i}/{len(test_samples)}: {sample['filename']}")
        print("-" * 40)
        
        audio_path = Path(training_data_dir) / "audio" / sample['filename']
        
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            continue
        
        try:
            # Run analysis
            features, predictions = inference_system.analyze_audio_file(audio_path)
            
            # Compare with ground truth
            ground_truth = sample['params']
            
            print("🎯 GROUND TRUTH vs PREDICTIONS:")
            print("-" * 30)
            
            for param in ground_truth.keys():
                if param in predictions:
                    gt_val = ground_truth[param]
                    pred_val = predictions[param]
                    
                    if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
                        error = abs(gt_val - pred_val)
                        error_pct = (error / (abs(gt_val) + 1e-10)) * 100
                        print(f"   {param}:")
                        print(f"     Truth: {gt_val:.4f}")
                        print(f"     Pred:  {pred_val:.4f}")
                        print(f"     Error: {error:.4f} ({error_pct:.1f}%)")
                    else:
                        match = "✅" if gt_val == pred_val else "❌"
                        print(f"   {param}: {gt_val} → {pred_val} {match}")
                else:
                    print(f"   {param}: {ground_truth[param]} → [NOT PREDICTED]")
            
            results.append({
                'sample': sample,
                'predictions': predictions,
                'ground_truth': ground_truth
            })
            
        except Exception as e:
            print(f"❌ Failed to analyze {sample['filename']}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Inference script for sample-to-synthesis')
    parser.add_argument('--audio', type=str, help='Path to audio file to analyze')
    parser.add_argument('--model_dir', type=str, default='trained_models', 
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                       help='Directory to save analysis results')
    parser.add_argument('--test_training', action='store_true',
                       help='Test with samples from training dataset')
    parser.add_argument('--training_dir', type=str, default='synth_training_data',
                       help='Training data directory')
    parser.add_argument('--frequency', type=float, default=440,
                       help='Frequency for synthesized audio (Hz)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration for synthesized audio (seconds)')
    
    args = parser.parse_args()
    
    # Initialize inference system
    try:
        inference = SynthInference(model_dir=args.model_dir)
    except Exception as e:
        print(f"\n💡 SOLUTION: Run the training pipeline first:")
        print("   1. python training_data_generation.py")
        print("   2. python audio_feature_extractor.py") 
        print("   3. python model.py")
        print("   4. Then run this inference script")
        return
    
    if args.test_training:
        # Test with training samples
        test_with_training_samples(inference, args.training_dir)
    
    elif args.audio:
        # Analyze specific audio file
        if not Path(args.audio).exists():
            print(f"❌ Audio file not found: {args.audio}")
            return
        
        inference.run_full_analysis(
            args.audio, 
            args.output_dir, 
            None, 
            args.duration
        )
    
    else:
        # Interactive mode - analyze some training samples by default
        print("🎵 No specific audio file provided. Testing with training samples...")
        test_with_training_samples(inference, args.training_dir, num_samples=3)

if __name__ == "__main__":
    main()