import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
import sys

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

spec = importlib.util.spec_from_file_location("inverse_design", os.path.join(os.path.dirname(__file__), "src", "inverse_design.py"))
inverse_design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inverse_design)

DataPreprocessor = data_preprocessing.DataPreprocessor
EnsembleModel = models.EnsembleModel
sMAPE = models.sMAPE
InverseDesigner = inverse_design.InverseDesigner

def calculate_weighted_error(y_true, y_pred):
    """Calculate the weighted error metric for evaluation."""
    mae_p80 = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_r95 = mean_absolute_error(y_true[:, 3], y_pred[:, 3])
    smape_fines = sMAPE(y_true[:, 1], y_pred[:, 1])
    smape_oversize = sMAPE(y_true[:, 2], y_pred[:, 2])
    mae_r50_fines = mean_absolute_error(y_true[:, 4], y_pred[:, 4])
    mae_r50_oversize = mean_absolute_error(y_true[:, 5], y_pred[:, 5])
    
    weighted_error = (0.30 * mae_p80 + 0.20 * mae_r95 + 
                     0.15 * smape_fines + 0.15 * smape_oversize +
                     0.10 * mae_r50_fines + 0.10 * mae_r50_oversize)
    
    return weighted_error

def main():
    """Main training and prediction pipeline."""
    print("🚀 Asteroid Impact Ejecta Prediction System")
    print("=" * 50)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    ensemble = EnsembleModel()
    
    # Data path
    data_path = "c:/Users/MSI 123/Downloads/Boom-Challenge-Datasets-main (1)/Boom-Challenge-Datasets-main"
    
    print("\n📊 Loading and preprocessing data...")
    # Prepare data
    (X_train_scaled, y_train_scaled, y_train_orig, X_test_scaled, 
     X_train_orig, y_train_orig, X_test_orig, input_cols, output_cols) = preprocessor.prepare_data(data_path)
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    print(f"Feature count: {len(input_cols)}")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )
    
    print("\n🤖 Training ensemble model...")
    # Train ensemble
    ensemble.fit(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # Validate model
    print("\n📈 Validating model...")
    val_pred_scaled = ensemble.predict(X_val_split)
    val_pred = preprocessor.scaler_output.inverse_transform(val_pred_scaled)
    val_true = preprocessor.scaler_output.inverse_transform(y_val_split)
    
    weighted_error = calculate_weighted_error(val_true, val_pred)
    score = 100 / (1 + weighted_error)
    
    print(f"Validation Weighted Error: {weighted_error:.4f}")
    print(f"Validation Score: {score:.2f}")
    
    # Individual metric breakdown
    mae_p80 = mean_absolute_error(val_true[:, 0], val_pred[:, 0])
    mae_r95 = mean_absolute_error(val_true[:, 3], val_pred[:, 3])
    smape_fines = sMAPE(val_true[:, 1], val_pred[:, 1])
    smape_oversize = sMAPE(val_true[:, 2], val_pred[:, 2])
    print(f"P80 MAE: {mae_p80:.2f}")
    print(f"R95 MAE: {mae_r95:.2f}")
    print(f"Fines sMAPE: {smape_fines:.2f}%")
    print(f"Oversize sMAPE: {smape_oversize:.2f}%")
    
    print("\n🎯 Generating test predictions...")
    # Make predictions on test set
    test_pred_scaled = ensemble.predict(X_test_scaled)
    test_pred = preprocessor.scaler_output.inverse_transform(test_pred_scaled)
    
    # Apply physics constraints
    test_pred_df = pd.DataFrame(test_pred, columns=output_cols)
    test_pred_df = preprocessor.apply_physics_constraints(test_pred_df)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'scenario_id': range(len(test_pred_df)),
        'P80': test_pred_df['P80'],
        'fines_frac': test_pred_df['fines_frac'],
        'oversize_frac': test_pred_df['oversize_frac'],
        'R95': test_pred_df['R95'],
        'R50_fines': test_pred_df['R50_fines'],
        'R50_oversize': test_pred_df['R50_oversize']
    })
    
    submission_df.to_csv('prediction_submission.csv', index=False)
    print(f"✅ Prediction submission saved: {len(submission_df)} scenarios")
    
    print("\n🔧 Inverse Design Optimization...")
    # Inverse design
    try:
        designer = InverseDesigner(ensemble, preprocessor, f"{data_path}/inverse_design/constraints.json")
        
        # Run optimization
        scenarios = designer.optimize(method='genetic')
        
        if scenarios:
            print(f"✅ Found {len(scenarios)} valid scenarios")
            
            # Create submission
            design_submission = designer.create_submission('design_submission.csv')
            
            # Print summary
            print("\n📊 Inverse Design Summary:")
            for i, scenario in enumerate(scenarios[:5]):  # Show first 5
                p80 = scenario['outputs'][0]
                r95 = scenario['outputs'][3]
                score = scenario['score']
                print(f"  Scenario {i}: P80={p80:.1f}, R95={r95:.1f}, Score={score:.4f}")
        else:
            print("⚠️  No valid scenarios found")
            
    except Exception as e:
        print(f"⚠️  Inverse design failed: {e}")
    
    print("\n💾 Saving models...")
    # Save models
    os.makedirs('models', exist_ok=True)
    # Save only tree-based models (skip neural network due to pickling issues)
    model_data = {
        'gb_model': ensemble.models['gb'],
        'lgb_model': ensemble.models['lgb'], 
        'xgb_model': ensemble.models['xgb'],
        'weights': {k: v for k, v in ensemble.weights.items() if k != 'nn'}
    }
    # Re-normalize weights for tree models only
    total_weight = sum(model_data['weights'].values())
    model_data['weights'] = {k: v/total_weight for k, v in model_data['weights'].items()}
    
    joblib.dump(model_data, 'models/ensemble_model.pkl')
    
    # Save preprocessor components separately
    preprocessor_data = {
        'scaler_input': preprocessor.scaler_input,
        'scaler_output': preprocessor.scaler_output,
        'feature_names': preprocessor.feature_names
    }
    joblib.dump(preprocessor_data, 'models/preprocessor.pkl')
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📁 Generated files:")
    print("  - prediction_submission.csv")
    print("  - design_submission.csv")
    print("  - models/ensemble_model.pkl")
    print("  - models/preprocessor.pkl")

if __name__ == "__main__":
    main()
