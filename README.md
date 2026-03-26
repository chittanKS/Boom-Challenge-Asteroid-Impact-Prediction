# 🚀 Boom Challenge: Asteroid Impact Ejecta Prediction

## 📋 Challenge Overview

This project implements a physics-informed machine learning system for the Boom Challenge, focusing on asteroid impact debris field prediction with two main components:

### 1️⃣ Forward Prediction (Mandatory)
- **Goal**: Predict 6 ejecta outcomes from 8 impact parameters
- **Input**: 8 impact parameters (energy, angle_rad, coupling, strength, porosity, gravity, atmosphere, shape_factor)
- **Output**: 6 ejecta outcomes (P80, fines_frac, oversize_frac, R95, R50_fines, R50_oversize)

### 2️⃣ Inverse Design (Optional - Bonus Points)
- **Goal**: Generate 20 valid impact scenarios satisfying constraints
- **Constraints**: 96 ≤ P80 ≤ 101 mm and R95 ≤ 175 m
- **Optimization**: Minimize energy and ejecta range for small impact score

---

## 📁 Essential Project Structure

```
Boom/
├── 📄 README.md                    # Complete project documentation
├── 📄 BOOM_CHALLENGE_PROJECT.md    # Comprehensive project summary
├── 📄 requirements.txt             # Python dependencies
├── 🐍 run_training.py            # Main training & prediction pipeline
├── 🐍 generate_inverse_design_robust.py # Inverse design optimization
├── 📁 src/                       # Core ML modules
│   ├── data_preprocessing.py    # Data loading & feature engineering
│   ├── models.py               # Ensemble ML models
│   └── inverse_design.py       # Optimization algorithms
├── 📁 models/                    # Saved model artifacts
│   ├── ensemble_model.pkl       # Trained ensemble models
│   └── preprocessor.pkl       # Data preprocessing components
├── 📊 prediction_submission.csv  # Forward predictions (492 scenarios)
└── 📊 design_submission.csv      # Inverse design (20 scenarios)
```

---

## 🔬 Technical Approach

### Data Preprocessing & Feature Engineering

**Physics-Inspired Features:**
- `energy_density = energy / strength`
- `gravity_scaled_energy = energy / gravity`
- `fragmentation_index = shape_factor × coupling`
- `porosity_effect = porosity × energy`
- `atmospheric_drag_factor = atmosphere × angle_rad`
- `impact_momentum = energy × angle_rad`
- `material_resistance = strength / (1 + porosity)`
- `gravitational_binding = gravity × porosity`

### Model Architecture

**Ensemble Approach:**
1. **Gradient Boosting** (scikit-learn)
2. **LightGBM** (optimized gradient boosting)
3. **XGBoost** (extreme gradient boosting)
4. **Physics-Informed Neural Network** (PyTorch)

**Custom Loss Function:**
```
Weighted Error = 0.30 × MAE(P80) 
               + 0.20 × MAE(R95)
               + 0.15 × sMAPE(fines_frac)
               + 0.15 × sMAPE(oversize_frac)
               + 0.10 × MAE(R50_fines)
               + 0.10 × MAE(R50_oversize)
```

**Physics Constraints:**
- `fines_frac + oversize_frac ≤ 1`
- `P80 > 0`
- `R95 ≥ R50_fines` and `R95 ≥ R50_oversize`

---

## 🚀 Step-by-Step Execution Guide

### 📋 Prerequisites
- Python 3.8+
- Dataset: `c:/Users/MSI 123/Downloads/Boom-Challenge-Datasets-main (1)/Boom-Challenge-Datasets-main/`

### ⚡ Quick Start

#### Step 1: Install Dependencies
```bash
cd "c:/Users/MSI 123/Boom"
pip install -r requirements.txt
```

#### Step 2: Run Training & Forward Prediction
```bash
& "C:\Users\MSI 123\AppData\Local\Programs\Python\Python311\python.exe" run_training.py
```

**Expected Output:**
```
🚀 Asteroid Impact Ejecta Prediction System
==================================================
📊 Loading and preprocessing data...
Training data shape: (2930, 18)
Test data shape: (492, 18)
🤖 Training ensemble model...
✓ Gradient Boosting trained
✓ LightGBM trained
✓ XGBoost trained
✓ Neural Network trained
📈 Validating model...
Validation Weighted Error: ~26.0
Validation Score: ~3.7/100
🎯 Generating test predictions...
✅ Prediction submission saved: 492 scenarios
🎉 Pipeline completed successfully!
```

#### Step 3: Generate Inverse Design Scenarios
```bash
& "C:\Users\MSI 123\AppData\Local\Programs\Python\Python311\python.exe" generate_inverse_design_robust.py
```

**Expected Output:**
```
🎯 Asteroid Impact Inverse Design Optimization
==================================================
📂 Loading trained models...
🔍 Finding valid scenarios...
✅ Found 20 valid scenarios!
📁 Design submission saved: design_submission.csv
📈 Total valid scenarios: 20
All scenarios satisfy constraints: ✅
```

---

## 📊 Expected Results

### Forward Prediction Performance
- **Validation Weighted Error**: ~26.0
- **Validation Score**: ~3.7/100
- **Test Predictions**: 492 scenarios
- **Individual Metrics**:
  - P80 MAE: ~12.2
  - R95 MAE: ~41.4
  - Fines sMAPE: ~20.5%
  - Oversize sMAPE: ~22.5%

### Inverse Design Results
- **Valid Scenarios**: 20/20
- **Constraint Satisfaction**: 100%
- **P80 Range**: 96.3-100.7 mm
- **R95 Range**: 149.2-174.3 m
- **Small Impact Scores**: 0.1251-0.3278

---

## 📁 Submission Files

### 1️⃣ Forward Prediction: `prediction_submission.csv`
```csv
scenario_id,P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize
0,145.77,0.1810,0.2416,919.99,823.94,395.68
1,124.61,0.0437,0.1490,233.77,233.77,106.61
...
491, ...
```

### 2️⃣ Inverse Design: `design_submission.csv`
```csv
submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor
0,0.6441,0.7606,1.6868,3.8283,0.0959,6.6173,0.2012,1.2506
1,0.8895,0.9621,1.4346,2.8596,0.0306,10.4699,0.0617,1.4775
...
19, ...
```

---

## 🎯 Key Innovations

1. **Physics-Informed Feature Engineering**: Domain knowledge embedded in feature creation
2. **Multi-Objective Ensemble**: Combines diverse model architectures with optimal weighting
3. **Constraint-Aware Training**: Physics penalties integrated into loss function
4. **Diversity-Preserving Optimization**: Ensures varied inverse design solutions
5. **Robust Validation**: Comprehensive evaluation with physics consistency checks

---

## 🏆 Challenge Success Criteria

### ✅ Forward Prediction (Mandatory)
- [x] 492 test predictions generated
- [x] All required columns present
- [x] scenario_id matches test index (0-based)
- [x] Physics constraints applied

### ✅ Inverse Design (Optional - Bonus)
- [x] 20 scenarios generated
- [x] All P80 values in [96, 101] range
- [x] All R95 values ≤ 175
- [x] All parameters within input bounds
- [x] Optimized for small impact score

---

## 🔧 Troubleshooting

### Common Issues:
1. **Module not found**: Ensure `requirements.txt` is installed
2. **Dataset path errors**: Verify dataset location in Downloads folder
3. **Memory issues**: Close other applications during training
4. **Permission errors**: Run PowerShell as Administrator

### Model Loading Issues:
```bash
# If models don't load, retrain first:
& "C:\Users\MSI 123\AppData\Local\Programs\Python\Python311\python.exe" run_training.py
```

---

## 📈 Performance Metrics

### Training Performance:
- **Training Time**: 5-10 minutes
- **Memory Usage**: ~2GB RAM
- **CPU Usage**: 80-100% (multi-core)

### Inverse Design Performance:
- **Optimization Time**: 2-5 minutes
- **Success Rate**: 100% constraint satisfaction
- **Diversity**: Minimum distance threshold enforced

---

## 🎉 Ready for Submission!

After completing both steps, you'll have:

1. ✅ **Forward Prediction**: `prediction_submission.csv` with 492 scenarios
2. ✅ **Inverse Design**: `design_submission.csv` with 20 optimized scenarios  
3. ✅ **Physics-Aware**: All predictions respect physical constraints
4. ✅ **Optimized**: Small impact scores minimized for bonus points

The project is now ready for Boom Challenge submission with both mandatory and optional components completed!

---

*Last Updated: March 2026*
*Dataset: Boom-Challenge-Datasets-main (1)*
*Python: 3.11*
