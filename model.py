import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

class SynthParameterPredictor:
    def __init__(self, model_dir="trained_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Store models and scalers
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = None
        
    def load_dataset(self, features_csv_path):
        """Load the features dataset"""
        df = pd.read_csv(features_csv_path)
        print(f"Loaded dataset: {df.shape}")
        
        # Separate features from targets
        target_cols = [col for col in df.columns if col.startswith('target_')]
        feature_cols = [col for col in df.columns if not col.startswith(('target_', 'sample_id', 'filename', 'frequency', 'duration'))]
        
        print(f"Features: {len(feature_cols)}")
        print(f"Targets: {len(target_cols)}")
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        return df, feature_cols, target_cols
    
    def preprocess_data(self, df, feature_cols, target_cols):
        """Preprocess features and targets"""
        print("\nPreprocessing data...")
        
        # Extract features and targets
        X = df[feature_cols].copy()
        y_dict = {}
        
        # Handle each target parameter
        for target_col in target_cols:
            param_name = target_col.replace('target_', '')
            y_dict[param_name] = df[target_col].copy()
        
        # Handle missing values in features
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            print(f"Filling {missing_before} missing feature values with median...")
            X = X.fillna(X.median())
        
        # Remove features with no variation
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            print(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            self.feature_names = X.columns.tolist()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"Final feature shape: {X.shape}")
        
        return X, y_dict
    
    def train_regression_model(self, X_train, X_test, y_train, y_test, param_name, model_type='random_forest'):
        """Train a regression model for a continuous parameter"""
        print(f"\nTraining regression model for {param_name}...")
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"  Train R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
        print(f"  Test R²:  {test_r2:.3f}, RMSE: {test_rmse:.3f}")
        
        # Store model and scaler
        self.models[param_name] = model
        self.scalers[param_name] = scaler
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 5 important features:")
            for _, row in feature_importance.head().iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")
        
        return {
            'model_type': model_type,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
        }
    
    def train_classification_model(self, X_train, X_test, y_train, y_test, param_name):
        """Train a classification model for categorical parameters"""
        print(f"\nTraining classification model for {param_name}...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train_encoded)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_acc = accuracy_score(y_train_encoded, y_pred_train)
        test_acc = accuracy_score(y_test_encoded, y_pred_test)
        
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy:  {test_acc:.3f}")
        
        # Classification report
        print(f"  Classification Report:")
        report = classification_report(y_test_encoded, y_pred_test, 
                                     target_names=label_encoder.classes_, 
                                     output_dict=True)
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"    {class_name}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")
        
        # Store model, scaler, and encoder
        self.models[param_name] = model
        self.scalers[param_name] = scaler
        self.label_encoders[param_name] = label_encoder
        
        return {
            'model_type': 'classification',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': report
        }
    
    def train_all_models(self, features_csv_path, test_size=0.2):
        """Train models for all parameters"""
        # Load data
        df, feature_cols, target_cols = self.load_dataset(features_csv_path)
        X, y_dict = self.preprocess_data(df, feature_cols, target_cols)
        
        # Split data
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        
        results = {}
        
        # Define parameter types
        categorical_params = ['osc_type', 'filter_type']
        continuous_params = [param for param in y_dict.keys() if param not in categorical_params]
        
        print(f"\nTraining models for {len(continuous_params)} continuous and {len(categorical_params)} categorical parameters...")
        
        # Train continuous parameter models
        for param_name in continuous_params:
            y_train = y_dict[param_name].loc[X_train.index]
            y_test = y_dict[param_name].loc[X_test.index]
            
            # Try both model types for continuous parameters
            rf_results = self.train_regression_model(X_train, X_test, y_train, y_test, param_name, 'random_forest')
            results[param_name] = rf_results
        
        # Train categorical parameter models
        for param_name in categorical_params:
            y_train = y_dict[param_name].loc[X_train.index]
            y_test = y_dict[param_name].loc[X_test.index]
            
            clf_results = self.train_classification_model(X_train, X_test, y_train, y_test, param_name)
            results[param_name] = clf_results
        
        # Save models
        self.save_models()
        
        return results
    
    def save_models(self):
        """Save all trained models and scalers"""
        print(f"\nSaving models to {self.model_dir}...")
        
        # Save models
        for param_name, model in self.models.items():
            model_path = self.model_dir / f"{param_name}_model.pkl"
            joblib.dump(model, model_path)
        
        # Save scalers
        for param_name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"{param_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
        
        # Save label encoders
        for param_name, encoder in self.label_encoders.items():
            encoder_path = self.model_dir / f"{param_name}_encoder.pkl"
            joblib.dump(encoder, encoder_path)
        
        # Save feature names
        feature_names_path = self.model_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load saved models"""
        print(f"Loading models from {self.model_dir}...")
        
        # Load feature names
        feature_names_path = self.model_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        # Load models
        for model_file in self.model_dir.glob("*_model.pkl"):
            param_name = model_file.stem.replace("_model", "")
            self.models[param_name] = joblib.load(model_file)
        
        # Load scalers
        for scaler_file in self.model_dir.glob("*_scaler.pkl"):
            param_name = scaler_file.stem.replace("_scaler", "")
            self.scalers[param_name] = joblib.load(scaler_file)
        
        # Load encoders
        for encoder_file in self.model_dir.glob("*_encoder.pkl"):
            param_name = encoder_file.stem.replace("_encoder", "")
            self.label_encoders[param_name] = joblib.load(encoder_file)
        
        print(f"Loaded {len(self.models)} models")
    
    def predict_parameters(self, audio_features):
        """Predict synthesis parameters from audio features"""
        if not self.models:
            raise ValueError("No models loaded. Train models first or load saved models.")
        
        # Convert to DataFrame if needed
        if isinstance(audio_features, dict):
            audio_features = pd.DataFrame([audio_features])
        
        # Ensure we have the right features
        missing_features = set(self.feature_names) - set(audio_features.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Fill with median values or zeros
            for feature in missing_features:
                audio_features[feature] = 0
        
        # Select only the features we trained on
        X = audio_features[self.feature_names]
        
        predictions = {}
        
        for param_name, model in self.models.items():
            # Scale features
            scaler = self.scalers[param_name]
            X_scaled = scaler.transform(X)
            
            # Predict
            pred = model.predict(X_scaled)
            
            # Handle categorical predictions
            if param_name in self.label_encoders:
                encoder = self.label_encoders[param_name]
                pred = encoder.inverse_transform(pred)
            
            predictions[param_name] = pred[0] if len(pred) == 1 else pred
        
        return predictions

# Main execution
if __name__ == "__main__":
    # Create trainer
    trainer = SynthParameterPredictor()
    
    # Train models
    print("Starting ML model training...")
    
    # Use the features CSV from the previous step
    features_csv = "synth_training_data/features.csv"
    
    # Train all models
    results = trainer.train_all_models(features_csv)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    for param_name, result in results.items():
        print(f"\n{param_name.upper()}:")
        if result['model_type'] == 'classification':
            print(f"  Type: Classification")
            print(f"  Test Accuracy: {result['test_accuracy']:.3f}")
        else:
            print(f"  Type: Regression")
            print(f"  Test R²: {result['test_r2']:.3f}")
            print(f"  Test RMSE: {result['test_rmse']:.3f}")
    
    print(f"\nModels saved to: {trainer.model_dir}")
    print("Ready for inference on new audio samples!")