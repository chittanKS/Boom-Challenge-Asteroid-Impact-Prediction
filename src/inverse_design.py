import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import optuna
import warnings
warnings.filterwarnings('ignore')

class InverseDesigner:
    """Inverse design optimization for asteroid impact scenarios."""
    
    def __init__(self, forward_model, preprocessor, constraints_path):
        self.forward_model = forward_model
        self.preprocessor = preprocessor
        self.constraints = self._load_constraints(constraints_path)
        self.best_scenarios = []
        
    def _load_constraints(self, constraints_path):
        """Load design constraints from JSON file."""
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
        return constraints
    
    def predict_outputs(self, input_features):
        """Predict outputs using the forward model."""
        # Ensure input_features has the same feature names as training data
        if hasattr(self.preprocessor, 'feature_names'):
            # Convert to DataFrame with correct column names
            if isinstance(input_features, np.ndarray):
                if input_features.ndim == 1:
                    input_features = input_features.reshape(1, -1)
                input_df = pd.DataFrame(input_features, columns=self.preprocessor.feature_names[:input_features.shape[1]])
            else:
                input_df = input_features
        else:
            input_df = input_features
            
        # Scale input features
        input_scaled = self.preprocessor.scaler_input.transform(input_df)
        
        # Make prediction
        pred_scaled = self.forward_model.predict(input_scaled)
        
        # Inverse scale to get actual values
        predictions = self.preprocessor.scaler_output.inverse_transform(pred_scaled)
        
        return predictions
    
    def is_valid_scenario(self, outputs):
        """Check if scenario meets design constraints."""
        p80, r95 = outputs[0], outputs[3]
        
        p80_valid = self.constraints['constraints']['p80_min'] <= p80 <= self.constraints['constraints']['p80_max']
        r95_valid = r95 <= self.constraints['constraints']['r95_max']
        
        return p80_valid and r95_valid
    
    def calculate_small_impact_score(self, inputs, outputs):
        """Calculate small impact score (lower is better)."""
        # Normalize inputs to [0, 1] range
        energy_norm = (inputs[0] - self.constraints['input_bounds']['energy']['min']) / \
                     (self.constraints['input_bounds']['energy']['max'] - self.constraints['input_bounds']['energy']['min'])
        
        # Score: prioritize low energy and low R95
        score = 0.7 * energy_norm + 0.3 * (outputs[3] / 500.0)  # Normalize R95 by typical max
        
        return score
    
    def random_search_optimization(self, n_candidates=10000):
        """Random search with filtering."""
        print("Running random search optimization...")
        
        bounds = self.constraints['input_bounds']
        valid_scenarios = []
        
        for i in range(n_candidates):
            # Generate random candidate
            candidate = {
                'energy': np.random.uniform(bounds['energy']['min'], bounds['energy']['max']),
                'angle_rad': np.random.uniform(bounds['angle_rad']['min'], bounds['angle_rad']['max']),
                'coupling': np.random.uniform(bounds['coupling']['min'], bounds['coupling']['max']),
                'strength': np.random.uniform(bounds['strength']['min'], bounds['strength']['max']),
                'porosity': np.random.uniform(bounds['porosity']['min'], bounds['porosity']['max']),
                'gravity': np.random.uniform(bounds['gravity']['min'], bounds['gravity']['max']),
                'atmosphere': np.random.uniform(bounds['atmosphere']['min'], bounds['atmosphere']['max']),
                'shape_factor': np.random.uniform(bounds['shape_factor']['min'], bounds['shape_factor']['max'])
            }
            
            # Convert to array for prediction
            input_features = np.array([list(candidate.values())])
            
            # Add physics features
            input_df = pd.DataFrame(input_features, columns=list(candidate.keys()))
            input_df = self.preprocessor.create_physics_features(input_df)
            
            # Predict outputs
            outputs = self.predict_outputs(input_df)[0]
            
            # Check validity
            if self.is_valid_scenario(outputs):
                score = self.calculate_small_impact_score(list(candidate.values()), outputs)
                valid_scenarios.append({
                    'inputs': candidate,
                    'outputs': outputs,
                    'score': score
                })
        
        # Sort by score and return top scenarios
        valid_scenarios.sort(key=lambda x: x['score'])
        return valid_scenarios[:20]
    
    def genetic_algorithm_optimization(self, population_size=100, generations=50):
        """Genetic algorithm for inverse design."""
        print("Running genetic algorithm optimization...")
        
        bounds = self.constraints['input_bounds']
        input_names = list(bounds.keys())
        
        def objective_function(params):
            """Objective function for optimization."""
            # Create input features
            input_features = params.reshape(1, -1)
            input_df = pd.DataFrame(input_features, columns=input_names)
            input_df = self.preprocessor.create_physics_features(input_df)
            
            # Predict outputs
            outputs = self.predict_outputs(input_df)[0]
            
            # Check validity
            if not self.is_valid_scenario(outputs):
                return 1000.0  # Large penalty for invalid scenarios
            
            # Calculate small impact score
            score = self.calculate_small_impact_score(params, outputs)
            return score
        
        # Genetic algorithm implementation
        def initialize_population(pop_size):
            population = []
            for _ in range(pop_size):
                individual = []
                for name in input_names:
                    individual.append(np.random.uniform(bounds[name]['min'], bounds[name]['max']))
                population.append(np.array(individual))
            return population
        
        def tournament_selection(population, fitness, k=3):
            selected = []
            for _ in range(len(population)):
                candidates = np.random.choice(len(population), k, replace=False)
                best_idx = candidates[np.argmin([fitness[i] for i in candidates])]
                selected.append(population[best_idx].copy())
            return selected
        
        def crossover(parent1, parent2):
            child1, child2 = parent1.copy(), parent2.copy()
            # Uniform crossover
            for i in range(len(child1)):
                if np.random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            return child1, child2
        
        def mutate(individual, mutation_rate=0.1):
            for i in range(len(individual)):
                if np.random.random() < mutation_rate:
                    name = input_names[i]
                    individual[i] = np.random.uniform(bounds[name]['min'], bounds[name]['max'])
            return individual
        
        # Initialize population
        population = initialize_population(population_size)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness = [objective_function(ind) for ind in population]
            
            # Selection
            selected = tournament_selection(population, fitness)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = crossover(selected[i], selected[i + 1])
                    new_population.extend([mutate(child1), mutate(child2)])
                else:
                    new_population.append(mutate(selected[i]))
            
            population = new_population[:population_size]
            
            # Print progress
            best_fitness = min(fitness)
            print(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
        
        # Get final solutions
        final_fitness = [objective_function(ind) for ind in population]
        
        # Extract valid scenarios
        valid_scenarios = []
        for ind, fit in zip(population, final_fitness):
            if fit < 1000:  # Valid scenario
                input_df = pd.DataFrame(ind.reshape(1, -1), columns=input_names)
                input_df = self.preprocessor.create_physics_features(input_df)
                outputs = self.predict_outputs(input_df)[0]
                
                candidate = {name: val for name, val in zip(input_names, ind)}
                valid_scenarios.append({
                    'inputs': candidate,
                    'outputs': outputs,
                    'score': fit
                })
        
        # Sort and return top 20
        valid_scenarios.sort(key=lambda x: x['score'])
        return valid_scenarios[:20]
    
    def ensure_diversity(self, scenarios, min_distance=0.1):
        """Ensure diversity in selected scenarios."""
        diverse_scenarios = []
        
        for scenario in scenarios:
            is_diverse = True
            for existing in diverse_scenarios:
                # Calculate Euclidean distance between input feature vectors
                dist = np.linalg.norm(
                    np.array(list(scenario['inputs'].values())) - 
                    np.array(list(existing['inputs'].values()))
                )
                if dist < min_distance:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_scenarios.append(scenario)
            
            if len(diverse_scenarios) >= 20:
                break
        
        return diverse_scenarios
    
    def optimize(self, method='genetic'):
        """Run inverse design optimization."""
        if method == 'random':
            scenarios = self.random_search_optimization()
        elif method == 'genetic':
            scenarios = self.genetic_algorithm_optimization()
        else:
            raise ValueError("Method must be 'random' or 'genetic'")
        
        # Ensure diversity
        diverse_scenarios = self.ensure_diversity(scenarios)
        
        # Pad with more scenarios if needed
        if len(diverse_scenarios) < 20:
            additional_scenarios = scenarios[len(diverse_scenarios):40]
            diverse_scenarios.extend(additional_scenarios)
        
        self.best_scenarios = diverse_scenarios[:20]
        return self.best_scenarios
    
    def create_submission(self, output_path):
        """Create submission file for inverse design."""
        if not self.best_scenarios:
            raise ValueError("No scenarios found. Run optimize() first.")
        
        submission_data = []
        for i, scenario in enumerate(self.best_scenarios):
            row = {'submission_id': i}
            row.update(scenario['inputs'])
            submission_data.append(row)
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        print(f"Design submission saved to {output_path}")
        
        return submission_df
