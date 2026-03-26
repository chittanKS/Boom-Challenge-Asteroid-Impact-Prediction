import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from file modules
import importlib.util

# Load modules directly
spec = importlib.util.spec_from_file_location("data_preprocessing", os.path.join(os.path.dirname(__file__), "src", "data_preprocessing.py"))
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

spec = importlib.util.spec_from_file_location("models", os.path.join(os.path.dirname(__file__), "src", "models.py"))
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

DataPreprocessor = data_preprocessing.DataPreprocessor
EnsembleModel = models.EnsembleModel

class SimpleEnsemble:
    """Simple ensemble for tree models only."""
    
    def __init__(self, model_data):
        self.models = {
            'gb': model_data['gb_model'],
            'lgb': model_data['lgb_model'],
            'xgb': model_data['xgb_model']
        }
        self.weights = model_data['weights']
    
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions['gb'])
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred

class RobustOptimizer:
    """Robust optimization for finding valid scenarios."""
    
    def __init__(self, model, preprocessor, constraints_path):
        self.model = model
        self.preprocessor = preprocessor
        self.constraints = self._load_constraints(constraints_path)
        self.valid_scenarios = []
        
    def _load_constraints(self, constraints_path):
        """Load design constraints."""
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
        return constraints
    
    def safe_create_features(self, df):
        """Safely create physics features with error handling."""
        df = df.copy()
        
        # Basic features with safe division
        df['energy_density'] = np.where(df['strength'] > 0.001, df['energy'] / df['strength'], 0)
        df['gravity_scaled_energy'] = np.where(df['gravity'] > 0.001, df['energy'] / df['gravity'], 0)
        df['porosity_effect'] = df['porosity'] * df['energy']
        df['fragmentation_index'] = df['shape_factor'] * df['coupling']
        df['atmospheric_drag_factor'] = df['atmosphere'] * df['angle_rad']
        df['impact_momentum'] = df['energy'] * df['angle_rad']
        
        # Safe material resistance
        df['material_resistance'] = np.where(
            (1 + df['porosity']) > 0.001,
            df['strength'] / (1 + df['porosity']),
            df['strength']
        )
        df['gravitational_binding'] = df['gravity'] * df['porosity']
        
        # Safe log transforms
        df['energy_log'] = np.log1p(np.maximum(df['energy'], 0))
        df['energy_density_log'] = np.log1p(np.maximum(df['energy_density'], 0))
        
        # Replace any inf or nan values
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        return df
    
    def predict_scenario(self, input_params):
        """Predict outcomes for a single scenario."""
        try:
            # Create DataFrame with correct column names
            input_df = pd.DataFrame([input_params], columns=self.preprocessor.feature_names[:len(input_params)])
            
            # Add physics features safely
            input_df = self.safe_create_features(input_df)
            
            # Ensure we have the right number of features
            if input_df.shape[1] != len(self.preprocessor.feature_names):
                # Pad or truncate to match expected features
                if input_df.shape[1] < len(self.preprocessor.feature_names):
                    # Add missing columns with zeros
                    for i in range(input_df.shape[1], len(self.preprocessor.feature_names)):
                        input_df[self.preprocessor.feature_names[i]] = 0
                else:
                    # Truncate extra columns
                    input_df = input_df.iloc[:, :len(self.preprocessor.feature_names)]
            
            # Scale and predict
            input_scaled = self.preprocessor.scaler_input.transform(input_df)
            pred_scaled = self.model.predict(input_scaled)
            predictions = self.preprocessor.scaler_output.inverse_transform(pred_scaled)
            
            return predictions[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return default values
            return [100, 0.1, 0.1, 200, 100, 50]
    
    def is_valid(self, outputs):
        """Check if scenario meets constraints."""
        try:
            p80, r95 = outputs[0], outputs[3]
            return (96 <= p80 <= 101) and (r95 <= 175)
        except:
            return False
    
    def calculate_score(self, inputs, outputs):
        """Calculate small impact score (lower is better)."""
        try:
            # Normalize energy to [0, 1]
            energy_norm = (inputs[0] - self.constraints['input_bounds']['energy']['min']) / \
                         (self.constraints['input_bounds']['energy']['max'] - self.constraints['input_bounds']['energy']['min'])
            
            # Normalize R95 (assume max ~500m)
            r95_norm = max(0, min(1, outputs[3] / 500.0))
            
            # Weighted score (prioritize low energy)
            score = 0.7 * energy_norm + 0.3 * r95_norm
            return score
        except:
            return 1.0
    
    def generate_smart_candidate(self):
        """Generate candidate with physics-informed heuristics."""
        bounds = self.constraints['input_bounds']
        
        # For small P80 (96-101mm), we want:
        # - Lower energy (smaller fragments)
        # - Higher strength (resists fragmentation)
        # - Higher gravity (shorter range)
        # - Moderate angle (not too steep)
        
        energy = np.random.uniform(bounds['energy']['min'], bounds['energy']['min'] + 1.5)
        angle = np.random.uniform(0.5, 1.2)  # Moderate angles
        coupling = np.random.uniform(bounds['coupling']['min'], bounds['coupling']['max'])
        strength = np.random.uniform(2.5, bounds['strength']['max'])  # Higher strength
        porosity = np.random.uniform(bounds['porosity']['min'], bounds['porosity']['max'])
        gravity = np.random.uniform(5.0, bounds['gravity']['max'])  # Higher gravity
        atmosphere = np.random.uniform(bounds['atmosphere']['min'], bounds['atmosphere']['max'])
        shape_factor = np.random.uniform(bounds['shape_factor']['min'], bounds['shape_factor']['max'])
        
        return [energy, angle, coupling, strength, porosity, gravity, atmosphere, shape_factor]
    
    def focused_search(self, n_candidates=50000):
        """Focused search with smart candidates."""
        print(f"Running focused search with {n_candidates} candidates...")
        
        valid_scenarios = []
        
        for i in tqdm(range(n_candidates), desc="Searching scenarios"):
            # Generate smart candidate
            candidate = self.generate_smart_candidate()
            
            # Predict outcomes
            outputs = self.predict_scenario(candidate)
            
            # Check validity
            if self.is_valid(outputs):
                score = self.calculate_score(candidate, outputs)
                valid_scenarios.append({
                    'inputs': candidate.copy(),
                    'outputs': outputs.copy(),
                    'score': score
                })
                
                print(f"✓ Valid scenario {len(valid_scenarios)}: P80={outputs[0]:.1f}, R95={outputs[3]:.1f}, Score={score:.4f}")
                
                if len(valid_scenarios) >= 25:  # Get extra for diversity selection
                    break
        
        return valid_scenarios
    
    def local_refinement(self, base_scenarios, iterations=1000):
        """Local refinement around promising scenarios."""
        print("Running local refinement...")
        
        refined_scenarios = base_scenarios.copy()
        
        for base_scenario in base_scenarios[:5]:  # Refine top 5
            current = base_scenario['inputs'].copy()
            
            for _ in range(iterations):
                # Small random perturbation
                bounds = self.constraints['input_bounds']
                perturbed = current.copy()
                
                # Perturb one parameter
                param_idx = random.randint(0, 7)
                param_name = list(bounds.keys())[param_idx]
                
                # Small perturbation
                param_range = bounds[param_name]['max'] - bounds[param_name]['min']
                perturbation = np.random.normal(0, 0.05 * param_range)
                
                perturbed[param_idx] = np.clip(
                    current[param_idx] + perturbation,
                    bounds[param_name]['min'],
                    bounds[param_name]['max']
                )
                
                # Evaluate
                outputs = self.predict_scenario(perturbed)
                
                if self.is_valid(outputs):
                    score = self.calculate_score(perturbed, outputs)
                    
                    new_scenario = {
                        'inputs': perturbed.copy(),
                        'outputs': outputs.copy(),
                        'score': score
                    }
                    
                    # Check if it's different enough
                    is_duplicate = False
                    for existing in refined_scenarios:
                        dist = np.linalg.norm(np.array(perturbed) - np.array(existing['inputs']))
                        if dist < 0.05:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        refined_scenarios.append(new_scenario)
                        
                        if len(refined_scenarios) >= 30:
                            return refined_scenarios
        
        return refined_scenarios
    
    def optimize(self):
        """Run optimization to find 20 valid scenarios."""
        # Strategy 1: Focused search
        scenarios = self.focused_search(30000)
        
        # Strategy 2: Local refinement if needed
        if len(scenarios) < 20:
            refined = self.local_refinement(scenarios, 500)
            scenarios.extend(refined)
        
        # Remove duplicates and ensure diversity
        unique_scenarios = []
        for scenario in scenarios:
            is_duplicate = False
            for existing in unique_scenarios:
                dist = np.linalg.norm(np.array(scenario['inputs']) - np.array(existing['inputs']))
                if dist < 0.1:  # Threshold for uniqueness
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_scenarios.append(scenario)
            
            if len(unique_scenarios) >= 20:
                break
        
        # Sort by score
        unique_scenarios.sort(key=lambda x: x['score'])
        return unique_scenarios[:20]
    
    def create_submission(self, scenarios, output_path):
        """Create submission file."""
        submission_data = []
        
        for i, scenario in enumerate(scenarios):
            row = {'submission_id': i}
            param_names = ['energy', 'angle_rad', 'coupling', 'strength', 
                          'porosity', 'gravity', 'atmosphere', 'shape_factor']
            
            for j, param_name in enumerate(param_names):
                row[param_name] = scenario['inputs'][j]
            
            submission_data.append(row)
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        print(f"Design submission saved to {output_path}")
        
        return submission_df

def main():
    """Main optimization pipeline."""
    print("🎯 Asteroid Impact Inverse Design Optimization")
    print("=" * 50)
    
    # Load trained models
    print("\n📂 Loading trained models...")
    try:
        model_data = joblib.load('models/ensemble_model.pkl')
        preprocessor_data = joblib.load('models/preprocessor.pkl')
    except FileNotFoundError:
        print("❌ Model files not found. Please run training first.")
        return
    
    # Reconstruct preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.scaler_input = preprocessor_data['scaler_input']
    preprocessor.scaler_output = preprocessor_data['scaler_output']
    preprocessor.feature_names = preprocessor_data['feature_names']
    
    # Reconstruct ensemble
    ensemble = SimpleEnsemble(model_data)
    
    # Initialize optimizer
    optimizer = RobustOptimizer(
        ensemble, 
        preprocessor, 
        "c:/Users/MSI 123/Downloads/Boom-Challenge-Datasets-main (1)/Boom-Challenge-Datasets-main/inverse_design/constraints.json"
    )
    
    print("\n🔍 Finding valid scenarios...")
    scenarios = optimizer.optimize()
    
    if len(scenarios) > 0:
        print(f"✅ Found {len(scenarios)} valid scenarios!")
        
        # Create submission
        submission_df = optimizer.create_submission(scenarios, 'design_submission.csv')
        
        # Print summary
        print("\n📊 All Scenarios:")
        for i, scenario in enumerate(scenarios):
            p80 = scenario['outputs'][0]
            r95 = scenario['outputs'][3]
            energy = scenario['inputs'][0]
            score = scenario['score']
            print(f"  Scenario {i}: P80={p80:.1f}mm, R95={r95:.1f}m, Energy={energy:.2f}, Score={score:.4f}")
        
        print(f"\n📁 Design submission saved: design_submission.csv")
        print(f"📈 Total valid scenarios: {len(scenarios)}")
        
        # Validate constraints
        print("\n🔍 Constraint Validation:")
        all_valid = True
        for scenario in scenarios:
            p80, r95 = scenario['outputs'][0], scenario['outputs'][3]
            if not (96 <= p80 <= 101 and r95 <= 175):
                all_valid = False
                break
        
        print(f"  All scenarios satisfy constraints: {'✅' if all_valid else '❌'}")
        
    else:
        print("❌ No valid scenarios found. Try adjusting optimization parameters.")

if __name__ == "__main__":
    main()
