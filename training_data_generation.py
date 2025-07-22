import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import random
from tqdm import tqdm
import soundfile as sf
from basic_synth import SubtractiveSynth, SynthParams

class TrainingDataGenerator:
    def __init__(self, output_dir="training_data", sample_rate=44100):
        self.output_dir = Path(output_dir)
        self.synth = SubtractiveSynth(sample_rate)
        self.sample_rate = sample_rate
        
        # Create output directories
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
    def random_params(self):
        """Generate random but musically reasonable synth parameters"""
        
        # Oscillator type - weighted toward more common ones
        osc_types = ['sine', 'saw', 'square', 'noise']
        osc_weights = [0.2, 0.4, 0.3, 0.1]  # favor saw and square
        osc_type = random.choices(osc_types, weights=osc_weights)[0]
        
        # Filter parameters
        # Cutoff: log distribution from 100Hz to 8000Hz (musical range)
        filter_cutoff = np.exp(random.uniform(np.log(100), np.log(8000)))
        
        # Resonance: mostly low values, occasionally high
        filter_resonance = random.choice([
            random.uniform(0.5, 2.0),  # 80% chance: subtle resonance
            random.uniform(0.5, 2.0),
            random.uniform(0.5, 2.0), 
            random.uniform(0.5, 2.0),
            random.uniform(3.0, 8.0)   # 20% chance: high resonance
        ])
        
        filter_type = random.choice(['lowpass', 'highpass', 'bandpass'])
        
        # ADSR envelope - ensure musical values
        attack = random.choice([
            random.uniform(0.001, 0.01),  # Fast attack (common)
            random.uniform(0.01, 0.1),    # Medium attack  
            random.uniform(0.1, 1.0)      # Slow attack (pads)
        ])
        
        decay = random.uniform(0.01, 0.5)
        sustain = random.uniform(0.1, 1.0)
        release = random.uniform(0.05, 2.0)
        
        # Amplitude
        amplitude = random.uniform(0.1, 0.5)  # Keep reasonable volume
        
        return SynthParams(
            osc_type=osc_type,
            osc_mix=1.0,  # Keep simple for now
            filter_cutoff=filter_cutoff,
            filter_resonance=filter_resonance,
            filter_type=filter_type,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            amplitude=amplitude
        )
    
    def random_note_params(self):
        """Generate random note parameters (frequency, duration)"""
        # Musical frequencies (roughly C2 to C6)
        midi_notes = range(36, 85)  # MIDI note numbers
        midi_note = random.choice(midi_notes)
        frequency = 440 * (2 ** ((midi_note - 69) / 12))  # Convert MIDI to Hz
        
        # Duration: mix of short and long notes
        duration = random.choice([
            random.uniform(0.5, 1.0),   # Short notes
            random.uniform(1.0, 2.0),   # Medium notes  
            random.uniform(2.0, 3.0)    # Long notes
        ])
        
        return frequency, duration
    
    def params_to_dict(self, params: SynthParams):
        """Convert SynthParams to dictionary for JSON serialization"""
        return {
            'osc_type': params.osc_type,
            'osc_mix': params.osc_mix,
            'filter_cutoff': params.filter_cutoff,
            'filter_resonance': params.filter_resonance,
            'filter_type': params.filter_type,
            'attack': params.attack,
            'decay': params.decay,
            'sustain': params.sustain,
            'release': params.release,
            'amplitude': params.amplitude
        }
    
    def generate_sample(self, sample_id):
        """Generate one training sample (audio + parameters)"""
        # Random parameters
        params = self.random_params()
        frequency, duration = self.random_note_params()
        
        # Generate audio
        try:
            audio = self.synth.synthesize(frequency, duration, params)
            
            # Save audio file
            audio_filename = f"sample_{sample_id:06d}.wav"
            audio_path = self.audio_dir / audio_filename
            sf.write(audio_path, audio, self.sample_rate)
            
            # Create metadata
            metadata = {
                'sample_id': sample_id,
                'filename': audio_filename,
                'frequency': frequency,
                'duration': duration,
                'sample_rate': self.sample_rate,
                'params': self.params_to_dict(params)
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            return None
    
    def generate_dataset(self, num_samples=5000, batch_size=100):
        """Generate full training dataset"""
        print(f"Generating {num_samples} training samples...")
        print(f"Output directory: {self.output_dir}")
        
        all_metadata = []
        
        # Generate in batches to save progress
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Batches"):
            batch_metadata = []
            
            batch_end = min(batch_start + batch_size, num_samples)
            for sample_id in range(batch_start, batch_end):
                metadata = self.generate_sample(sample_id)
                if metadata:
                    batch_metadata.append(metadata)
            
            all_metadata.extend(batch_metadata)
            
            # Save progress periodically
            if len(all_metadata) % (batch_size * 5) == 0:
                self.save_metadata(all_metadata, suffix=f"_partial_{len(all_metadata)}")
        
        # Save final metadata
        self.save_metadata(all_metadata)
        
        print(f"Successfully generated {len(all_metadata)} samples")
        print(f"Audio files saved to: {self.audio_dir}")
        print(f"Metadata saved to: {self.output_dir / 'metadata.json'}")
        
        return all_metadata
    
    def save_metadata(self, metadata_list, suffix=""):
        """Save metadata to JSON and CSV files"""
        # Save as JSON (complete data)
        json_path = self.output_dir / f"metadata{suffix}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save as CSV (flattened for easy analysis)
        if metadata_list:
            csv_data = []
            for item in metadata_list:
                row = {
                    'sample_id': item['sample_id'],
                    'filename': item['filename'],
                    'frequency': item['frequency'],
                    'duration': item['duration']
                }
                # Flatten parameters
                for key, value in item['params'].items():
                    row[f'param_{key}'] = value
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / f"metadata{suffix}.csv"
            df.to_csv(csv_path, index=False)
    
    def analyze_dataset(self, metadata_file="metadata.json"):
        """Analyze the generated dataset"""
        metadata_path = self.output_dir / metadata_file
        
        if not metadata_path.exists():
            print(f"Metadata file not found: {metadata_path}")
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nDataset Analysis:")
        print(f"Total samples: {len(metadata)}")
        
        # Analyze parameter distributions
        params_data = [item['params'] for item in metadata]
        df = pd.DataFrame(params_data)
        
        print(f"\nParameter distributions:")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                print(f"{col}: {df[col].min():.3f} - {df[col].max():.3f} (mean: {df[col].mean():.3f})")
            else:
                print(f"{col}: {df[col].value_counts().to_dict()}")

# Main execution
if __name__ == "__main__":
    # Create generator
    generator = TrainingDataGenerator(output_dir="synth_training_data")
    
    # Generate dataset - start with smaller number for testing
    print("Starting training data generation...")
    
    # Start with 1000 samples for testing, increase later
    metadata = generator.generate_dataset(num_samples=5000, batch_size=50)
    
    # Analyze the generated dataset
    generator.analyze_dataset()
    
    print("\nTraining data generation complete!")
    print("Next steps:")
    print("1. Listen to some samples to verify quality")
    print("2. If good, generate more samples (5000-10000)")
    print("3. Move to audio feature extraction")