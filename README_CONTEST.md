# 🚀 Boom Challenge: Asteroid Impact Ejecta Prediction

## 📋 Contest Overview
This project implements a physics-informed machine learning system for the Boom Challenge, focusing on asteroid impact debris field prediction with both mandatory forward prediction and optional inverse design components.

## 🎯 Challenge Components

### 1️⃣ Forward Prediction (Mandatory)
- **Goal**: Predict 6 ejecta outcomes from 8 impact parameters
- **Input**: 8 impact parameters (energy, angle_rad, coupling, strength, porosity, gravity, atmosphere, shape_factor)
- **Output**: 6 ejecta outcomes (P80, fines_frac, oversize_frac, R95, R50_fines, R50_oversize)
- **Results**: 492 test predictions in `prediction_submission.csv`

### 2️⃣ Inverse Design (Optional - Bonus)
- **Goal**: Generate 20 impact scenarios satisfying constraints
- **Constraints**: 96 ≤ P80 ≤ 101 mm and R95 ≤ 175 m
- **Results**: 20/20 valid scenarios in `design_submission.csv`

## 🏆 Results Summary

### Forward Prediction Performance
- **Validation Weighted Error**: 26.49
- **Validation Score**: 3.7/100
- **P80 MAE**: 12.58
- **R95 MAE**: 41.39

### Inverse Design Results
- **Success Rate**: 100% (20/20 scenarios)
- **P80 Range**: 96.7 - 100.8 mm ✅
- **R95 Range**: 142.8 - 174.4 m ✅
- **Best Small Impact Score**: 0.1213

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Dataset: Boom-Challenge-Datasets-main (1)

### Installation
```bash
pip install -r requirements.txt
```

### Run Project
```bash
# Training & Forward Prediction
python run_training.py

# Inverse Design Optimization
python generate_inverse_design_robust.py
```

## 📁 Project Structure
```
Boom/
├── 📄 README.md                    # This file
├── 📄 BOOM_CHALLENGE_PROJECT.md    # Complete technical summary
├── 📄 requirements.txt             # Dependencies
├── 🐍 run_training.py            # Main training pipeline
├── 🐍 generate_inverse_design_robust.py # Inverse design
├── 📁 src/                       # Core ML modules
├── 📁 models/                    # Saved model artifacts
├── 📊 prediction_submission.csv  # Forward predictions (492 scenarios)
└── 📊 design_submission.csv      # Inverse design (20 scenarios)
```

## 🔬 Technical Approach

### Physics-Informed Features
- Energy density, gravity scaling, fragmentation index
- Porosity effects, atmospheric drag, impact momentum
- Material resistance, gravitational binding

### Model Architecture
- **Ensemble**: Gradient Boosting, LightGBM, XGBoost, Neural Network
- **Custom Loss**: Weighted MAE + sMAPE + physics constraints
- **Optimization**: Genetic algorithm with diversity preservation

## 📊 Submission Files

### Forward Prediction: `prediction_submission.csv`
```csv
scenario_id,P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize
0,145.77,0.1810,0.2416,919.99,823.94,395.68
...
491,...
```

### Inverse Design: `design_submission.csv`
```csv
submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor
0,0.607,0.961,1.395,3.265,0.135,8.331,0.299,1.332
...
19,2.196,0.659,1.565,3.266,0.200,5.923,0.540,1.021
```

## 🎯 Constraint Validation

### P80 Constraint (96-101 mm)
- ✅ Range: 96.7 - 100.8 mm
- ✅ All 20 scenarios within range

### R95 Constraint (≤175 m)
- ✅ Maximum: 174.4 m
- ✅ All 20 scenarios under limit

## 🏅 Contest Readiness

- ✅ **Mandatory Component**: Forward prediction complete
- ✅ **Optional Component**: Inverse design complete
- ✅ **Constraint Satisfaction**: 100%
- ✅ **Submission Format**: Correct and validated
- ✅ **Documentation**: Complete

## 📚 Dataset

**Used**: Boom-Challenge-Datasets-main (1)
- Training: 2,930 scenarios
- Test: 492 scenarios
- Constraints: From inverse_design/constraints.json

## 👨‍💻 Authors

**Developed for Boom Challenge 2026**
- Physics-informed machine learning approach
- Ensemble model with custom loss function
- Genetic algorithm optimization for inverse design

---

**Status**: ✅ Ready for Boom Challenge submission! 🚀
