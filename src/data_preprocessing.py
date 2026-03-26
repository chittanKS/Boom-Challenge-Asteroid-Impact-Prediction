import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handle data loading, preprocessing, and feature engineering for asteroid impact prediction."""
    
    def __init__(self):
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.feature_names = None
        
    def load_data(self, data_path):
        """Load training data and labels."""
        train_df = pd.read_csv(f"{data_path}/forward_prediction/train.csv")
        train_labels = pd.read_csv(f"{data_path}/forward_prediction/train_labels.csv")
        test_df = pd.read_csv(f"{data_path}/forward_prediction/test.csv")
        
        # Merge training data with labels
        train_full = pd.concat([train_df, train_labels], axis=1)
        
        return train_full, test_df
    
    def create_physics_features(self, df):
        """Create physics-inspired features."""
        df = df.copy()
        
        # Energy-based features
        df['energy_density'] = df['energy'] / df['strength']
        df['gravity_scaled_energy'] = df['energy'] / df['gravity']
        df['porosity_effect'] = df['porosity'] * df['energy']
        
        # Fragmentation features
        df['fragmentation_index'] = df['shape_factor'] * df['coupling']
        
        # Atmospheric effects
        df['atmospheric_drag_factor'] = df['atmosphere'] * df['angle_rad']
        
        # Additional physics-informed features
        df['impact_momentum'] = df['energy'] * df['angle_rad']
        df['material_resistance'] = df['strength'] / (1 + df['porosity'])
        df['gravitational_binding'] = df['gravity'] * df['porosity']
        
        # Log transforms for skewed variables
        df['energy_log'] = np.log1p(df['energy'])
        df['energy_density_log'] = np.log1p(df['energy_density'])
        
        return df
    
    def apply_physics_constraints(self, predictions):
        """Apply physics constraints to predictions."""
        predictions = predictions.copy()
        
        # Ensure fines_frac + oversize_frac <= 1
        total_frac = predictions['fines_frac'] + predictions['oversize_frac']
        mask = total_frac > 1.0
        if mask.any():
            predictions.loc[mask, 'fines_frac'] = predictions.loc[mask, 'fines_frac'] / total_frac[mask]
            predictions.loc[mask, 'oversize_frac'] = predictions.loc[mask, 'oversize_frac'] / total_frac[mask]
        
        # Ensure P80 > 0
        predictions['P80'] = np.maximum(predictions['P80'], 0.1)
        
        # Ensure R95 >= R50_fines and R95 >= R50_oversize
        predictions['R95'] = np.maximum(predictions['R95'], predictions['R50_fines'])
        predictions['R95'] = np.maximum(predictions['R95'], predictions['R50_oversize'])
        
        return predictions
    
    def prepare_data(self, data_path):
        """Complete data preparation pipeline."""
        # Load data
        train_df, test_df = self.load_data(data_path)
        
        # Create physics features
        train_df = self.create_physics_features(train_df)
        test_df = self.create_physics_features(test_df)
        
        # Define input and output columns
        input_cols = [col for col in train_df.columns if col not in [
            'P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize'
        ]]
        output_cols = ['P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize']
        
        self.feature_names = input_cols
        
        # Split features and targets
        X_train = train_df[input_cols]
        y_train = train_df[output_cols]
        X_test = test_df[input_cols]
        
        # Scale features
        X_train_scaled = self.scaler_input.fit_transform(X_train)
        X_test_scaled = self.scaler_input.transform(X_test)
        
        # Scale targets (for neural network training)
        y_train_scaled = self.scaler_output.fit_transform(y_train)
        
        return (X_train_scaled, y_train_scaled, y_train, X_test_scaled, 
                X_train, y_train, X_test, input_cols, output_cols)
    
    def inverse_scale_predictions(self, predictions_scaled):
        """Convert scaled predictions back to original scale."""
        return self.scaler_output.inverse_transform(predictions_scaled)
