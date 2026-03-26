import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AsteroidDataset(Dataset):
    """PyTorch dataset for asteroid impact data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PhysicsInformedNN(nn.Module):
    """Physics-informed neural network for asteroid impact prediction."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super(PhysicsInformedNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CustomLoss(nn.Module):
    """Custom loss function with weighted MAE and physics constraints."""
    
    def __init__(self, weights=None):
        super(CustomLoss, self).__init__()
        if weights is None:
            # Default weights from specification
            self.weights = torch.tensor([0.30, 0.15, 0.15, 0.20, 0.10, 0.10])
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, predictions, targets):
        # MAE loss
        mae_loss = torch.mean(torch.abs(predictions - targets), dim=0)
        weighted_loss = torch.sum(mae_loss * self.weights)
        
        # Physics constraints as penalties
        physics_penalty = 0.0
        
        # fines_frac + oversize_frac <= 1
        frac_sum = predictions[:, 1] + predictions[:, 2]
        physics_penalty += torch.mean(torch.relu(frac_sum - 1.0)) * 10.0
        
        # P80 > 0
        physics_penalty += torch.mean(torch.relu(-predictions[:, 0])) * 5.0
        
        # R95 >= R50_fines and R95 >= R50_oversize
        physics_penalty += torch.mean(torch.relu(predictions[:, 3] - predictions[:, 4])) * 5.0
        physics_penalty += torch.mean(torch.relu(predictions[:, 3] - predictions[:, 5])) * 5.0
        
        return weighted_loss + physics_penalty

def sMAPE(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

class EnsembleModel:
    """Ensemble of models for robust prediction."""
    
    def __init__(self):
        self.models = {}
        self.weights = None
        
    def train_gb_model(self, X_train, y_train):
        """Train Gradient Boosting model."""
        gb_model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        )
        gb_model.fit(X_train, y_train)
        return gb_model
    
    def train_lgb_model(self, X_train, y_train):
        """Train LightGBM model."""
        lgb_model = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        )
        lgb_model.fit(X_train, y_train)
        return lgb_model
    
    def train_xgb_model(self, X_train, y_train):
        """Train XGBoost model."""
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model
    
    def train_nn_model(self, X_train, y_train, input_dim, output_dim, 
                      X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train Physics-Informed Neural Network."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create datasets
        train_dataset = AsteroidDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = AsteroidDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = PhysicsInformedNN(input_dim, output_dim).to(device)
        criterion = CustomLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if X_val is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                scheduler.step(val_loss / len(val_loader))
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        model.load_state_dict(best_model_state)
                        break
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble."""
        print("Training ensemble models...")
        
        # Train tree-based models
        self.models['gb'] = self.train_gb_model(X_train, y_train)
        print("✓ Gradient Boosting trained")
        
        self.models['lgb'] = self.train_lgb_model(X_train, y_train)
        print("✓ LightGBM trained")
        
        self.models['xgb'] = self.train_xgb_model(X_train, y_train)
        print("✓ XGBoost trained")
        
        # Train neural network
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        self.models['nn'] = self.train_nn_model(
            X_train, y_train, input_dim, output_dim, X_val, y_val
        )
        print("✓ Neural Network trained")
        
        # Calculate optimal weights using validation set
        if X_val is not None:
            self._calculate_optimal_weights(X_val, y_val)
        else:
            # Default equal weights
            self.weights = {name: 0.25 for name in self.models.keys()}
    
    def _calculate_optimal_weights(self, X_val, y_val):
        """Calculate optimal weights for ensemble based on validation performance."""
        predictions = {}
        errors = {}
        
        for name, model in self.models.items():
            if name == 'nn':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(device)
                    pred = model(X_val_tensor).cpu().numpy()
            else:
                pred = model.predict(X_val)
            
            predictions[name] = pred
            # Calculate weighted error for this model
            errors[name] = self._calculate_weighted_error(y_val, pred)
        
        # Inverse error weighting (lower error = higher weight)
        inv_errors = {name: 1.0 / error for name, error in errors.items()}
        total_inv_error = sum(inv_errors.values())
        self.weights = {name: inv_error / total_inv_error 
                       for name, inv_error in inv_errors.items()}
        
        print("Ensemble weights:", self.weights)
    
    def _calculate_weighted_error(self, y_true, y_pred):
        """Calculate the weighted error metric."""
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
    
    def predict(self, X):
        """Make predictions using the ensemble."""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'nn':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    pred = model(X_tensor).cpu().numpy()
            else:
                pred = model.predict(X)
            
            predictions[name] = pred
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions['gb'])
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
