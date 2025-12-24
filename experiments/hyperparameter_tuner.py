import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import optuna
from optuna.trial import Trial
import warnings
from datetime import datetime
import json
import pandas as pd
class HyperparameterTuner:
    """
    Advanced hyperparameter tuning with multiple strategies
    """
    
    def __init__(self,
                 search_space: Dict,
                 optimization_direction: str = 'maximize',
                 n_trials: int = 100):
        self.search_space = search_space
        self.direction = optimization_direction
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def tune(self,
             objective_function: callable,
             sampler_type: str = 'tpe',
             pruner_type: str = 'median') -> Tuple[Dict, float]:
        """
        Tune hyperparameters using specified strategy
        """
        # Create study
        sampler = self._create_sampler(sampler_type)
        pruner = self._create_pruner(pruner_type)
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"patchvision_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Define objective with search space
        def objective(trial: Trial) -> float:
            params = self._suggest_parameters(trial)
            return objective_function(params)
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return self.best_params, self.best_value
    
    def tune_with_warm_start(self,
                            objective_function: callable,
                            initial_params: Dict,
                            exploration_factor: float = 0.3) -> Tuple[Dict, float]:
        """
        Tune with warm start from initial parameters
        """
        # Create custom sampler that starts from initial params
        class WarmStartSampler(optuna.samplers.TPESampler):
            def __init__(self, initial_params, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.initial_params = initial_params
                
            def sample_relative(self, study, trial, search_space):
                if trial.number == 0:
                    return self.initial_params
                return super().sample_relative(study, trial, search_space)
        
        sampler = WarmStartSampler(initial_params)
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler
        )
        
        def objective(trial: Trial) -> float:
            if trial.number == 0:
                params = initial_params
            else:
                params = self._suggest_parameters_with_exploration(
                    trial, initial_params, exploration_factor
                )
            return objective_function(params)
        
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return self.best_params, self.best_value
    
    def multi_objective_tuning(self,
                              objective_functions: List[callable],
                              objectives_names: List[str]) -> List[Dict]:
        """
        Multi-objective hyperparameter tuning
        """
        study = optuna.create_study(
            directions=['maximize'] * len(objective_functions),
            study_name="multi_objective_tuning"
        )
        
        def multi_objective(trial: Trial):
            params = self._suggest_parameters(trial)
            return [func(params) for func in objective_functions]
        
        study.optimize(multi_objective, n_trials=self.n_trials)
        
        # Get Pareto front
        pareto_front = study.best_trials
        
        results = []
        for trial in pareto_front:
            results.append({
                'params': trial.params,
                'values': trial.values,
                'objectives': objectives_names
            })
            
        return results
    
    def analyze_parameter_importance(self) -> pd.DataFrame:
        """
        Analyze parameter importance
        """
        if self.study is None:
            raise ValueError("Run tuning first")
            
        importance = optuna.importance.get_param_importances(self.study)
        
        df = pd.DataFrame({
            'parameter': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return df
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history
        """
        if self.study is None:
            raise ValueError("Run tuning first")
            
        history = []
        for trial in self.study.trials:
            history.append({
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_complete
            })
            
        return pd.DataFrame(history)
    
    def save_study(self, filename: str):
        """
        Save study for later analysis
        """
        if self.study is None:
            raise ValueError("No study to save")
            
        # Save trials
        trials = []
        for trial in self.study.trials:
            trials.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
            
        study_data = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'trials': trials,
            'search_space': self.search_space
        }
        
        with open(filename, 'w') as f:
            json.dump(study_data, f, indent=2)
    
    # Helper methods
    def _suggest_parameters(self, trial: Trial) -> Dict:
        """Suggest parameters based on search space"""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'discrete':
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    param_config['q']
                )
                
        return params
    
    def _suggest_parameters_with_exploration(self,
                                           trial: Trial,
                                           baseline_params: Dict,
                                           exploration_factor: float) -> Dict:
        """Suggest parameters with exploration around baseline"""
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            baseline = baseline_params.get(param_name)
            
            if baseline is None:
                # Use normal suggestion
                params[param_name] = self._suggest_single_parameter(
                    trial, param_name, param_config
                )
            else:
                # Explore around baseline
                if param_type == 'float':
                    low = max(param_config['low'], 
                             baseline * (1 - exploration_factor))
                    high = min(param_config['high'], 
                              baseline * (1 + exploration_factor))
                    params[param_name] = trial.suggest_float(
                        param_name, low, high
                    )
                elif param_type == 'int':
                    low = max(param_config['low'],
                             int(baseline * (1 - exploration_factor)))
                    high = min(param_config['high'],
                              int(baseline * (1 + exploration_factor)))
                    params[param_name] = trial.suggest_int(
                        param_name, low, high
                    )
                else:
                    # For categorical, use normal suggestion
                    params[param_name] = self._suggest_single_parameter(
                        trial, param_name, param_config
                    )
                    
        return params
    
    @staticmethod
    def _suggest_single_parameter(trial: Trial,
                                 param_name: str,
                                 param_config: Dict):
        """Suggest single parameter"""
        param_type = param_config['type']
        
        if param_type == 'float':
            return trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=param_config.get('log', False)
            )
        elif param_type == 'int':
            return trial.suggest_int(
                param_name,
                param_config['low'],
                param_config['high'],
                log=param_config.get('log', False)
            )
        elif param_type == 'categorical':
            return trial.suggest_categorical(
                param_name,
                param_config['choices']
            )
    
    @staticmethod
    def _create_sampler(sampler_type: str):
        """Create sampler based on type"""
        if sampler_type == 'tpe':
            return optuna.samplers.TPESampler()
        elif sampler_type == 'random':
            return optuna.samplers.RandomSampler()
        elif sampler_type == 'cmaes':
            return optuna.samplers.CmaEsSampler()
        elif sampler_type == 'nsgaii':
            return optuna.samplers.NSGAIISampler()
        else:
            warnings.warn(f"Unknown sampler type: {sampler_type}, using TPE")
            return optuna.samplers.TPESampler()
    
    @staticmethod
    def _create_pruner(pruner_type: str):
        """Create pruner based on type"""
        if pruner_type == 'median':
            return optuna.pruners.MedianPruner()
        elif pruner_type == 'percentile':
            return optuna.pruners.PercentilePruner()
        elif pruner_type == 'hyperband':
            return optuna.pruners.HyperbandPruner()
        elif pruner_type == 'none':
            return optuna.pruners.NopPruner()
        else:
            warnings.warn(f"Unknown pruner type: {pruner_type}, using Median")
            return optuna.pruners.MedianPruner()