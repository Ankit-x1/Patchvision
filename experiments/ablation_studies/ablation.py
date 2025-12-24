import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import pandas as pd

class AblationStudy:
    """
    Systematic ablation studies for model components
    """
    
    def __init__(self, 
                 model_components: Dict[str, Any],
                 evaluation_metrics: List[str]):
        self.components = model_components
        self.metrics = evaluation_metrics
        self.results = {}
        self.baseline_performance = None
        
    def run_study(self,
                 test_data: Dict,
                 baseline_model: callable) -> Dict:
        """
        Run comprehensive ablation study
        """
        # Get baseline performance
        print("Running baseline evaluation...")
        self.baseline_performance = self._evaluate_model(
            baseline_model, test_data
        )
        
        # Ablate each component
        for component_name, component in self.components.items():
            print(f"Ablating component: {component_name}")
            
            # Create ablated model
            ablated_model = self._create_ablated_model(
                baseline_model, component_name
            )
            
            # Evaluate ablated model
            performance = self._evaluate_model(ablated_model, test_data)
            
            # Calculate performance drop
            performance_drop = {}
            for metric in self.metrics:
                baseline_val = self.baseline_performance.get(metric, 0)
                ablated_val = performance.get(metric, 0)
                drop = baseline_val - ablated_val
                performance_drop[metric] = drop
                
            self.results[component_name] = {
                'performance': performance,
                'performance_drop': performance_drop,
                'importance_score': self._calculate_importance(performance_drop)
            }
            
        return self.results
    
    def analyze_importance(self) -> pd.DataFrame:
        """
        Analyze component importance
        """
        importance_data = []
        
        for component, result in self.results.items():
            importance_data.append({
                'component': component,
                'importance_score': result['importance_score'],
                **result['performance_drop']
            })
            
        df = pd.DataFrame(importance_data)
        return df.sort_values('importance_score', ascending=False)
    
    def generate_report(self, 
                       output_file: Optional[str] = None) -> str:
        """
        Generate ablation study report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_performance': self.baseline_performance,
            'components_studied': list(self.components.keys()),
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        report_json = json.dumps(report, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)
                
        return report_json
    
    def _evaluate_model(self, 
                       model: callable, 
                       test_data: Dict) -> Dict:
        """
        Evaluate model on test data
        """
        # This is a placeholder - implement actual evaluation
        metrics = {}
        
        for metric in self.metrics:
            if metric == 'accuracy':
                # Simulate accuracy calculation
                metrics[metric] = np.random.random()
            elif metric == 'latency':
                # Simulate latency measurement
                metrics[metric] = np.random.random() * 100
            elif metric == 'memory':
                # Simulate memory usage
                metrics[metric] = np.random.random() * 1024
                
        return metrics
    
    def _create_ablated_model(self,
                             base_model: callable,
                             component_to_remove: str) -> callable:
        """
        Create model with specific component removed
        """
        # This is a placeholder - implement actual ablation
        # In practice, this would modify the model architecture
        return base_model
    
    @staticmethod
    def _calculate_importance(performance_drop: Dict) -> float:
        """
        Calculate component importance score
        """
        weights = {
            'accuracy': 0.5,
            'latency': 0.3,
            'memory': 0.2
        }
        
        score = 0
        for metric, drop in performance_drop.items():
            weight = weights.get(metric, 0.1)
            score += drop * weight
            
        return score
    
    def _generate_summary(self) -> Dict:
        """
        Generate study summary
        """
        if not self.results:
            return {}
            
        # Find most critical components
        critical = []
        for component, result in self.results.items():
            if result['importance_score'] > 0.1:  # Threshold
                critical.append({
                    'component': component,
                    'importance': result['importance_score']
                })
                
        # Sort by importance
        critical.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'total_components': len(self.components),
            'critical_components': critical[:3],  # Top 3
            'average_performance_drop': np.mean([
                r['importance_score'] for r in self.results.values()
            ])
        }

class ComponentAnalyzer:
    """
    Analyze individual component performance
    """
    
    def __init__(self):
        self.component_metrics = {}
        
    def analyze_component(self,
                         component: callable,
                         test_inputs: List,
                         component_name: str = "unknown") -> Dict:
        """
        Analyze specific component
        """
        metrics = {
            'latency': [],
            'memory': [],
            'output_variance': [],
            'numerical_stability': []
        }
        
        for test_input in test_inputs:
            # Measure latency
            import time
            start = time.time()
            output = component(test_input)
            latency = (time.time() - start) * 1000  # ms
            metrics['latency'].append(latency)
            
            # Measure memory (approximate)
            if hasattr(output, 'nbytes'):
                memory = output.nbytes / 1024 / 1024  # MB
                metrics['memory'].append(memory)
                
            # Store output for variance analysis
            metrics['output_variance'].append(output)
            
        # Calculate statistics
        stats = {}
        for metric_name, values in metrics.items():
            if values:
                stats[f'{metric_name}_mean'] = np.mean(values)
                stats[f'{metric_name}_std'] = np.std(values)
                stats[f'{metric_name}_min'] = np.min(values)
                stats[f'{metric_name}_max'] = np.max(values)
                
        # Calculate numerical stability
        if metrics['output_variance']:
            outputs = np.array(metrics['output_variance'])
            stats['numerical_stability'] = np.std(outputs) / (np.mean(np.abs(outputs)) + 1e-8)
            
        self.component_metrics[component_name] = stats
        return stats
    
    def compare_components(self,
                          component_dict: Dict[str, callable],
                          test_inputs: List) -> pd.DataFrame:
        """
        Compare multiple components
        """
        comparison_data = []
        
        for name, component in component_dict.items():
            metrics = self.analyze_component(component, test_inputs, name)
            metrics['component'] = name
            comparison_data.append(metrics)
            
        return pd.DataFrame(comparison_data)