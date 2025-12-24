import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from pathlib import Path
import hashlib

class ResultsManager:
    """
    Comprehensive results management for experiments
    """
    
    def __init__(self, 
                 results_dir: str = "results",
                 auto_version: bool = True):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        if auto_version:
            self.current_version = self._generate_version()
        else:
            self.current_version = "latest"
            
        self.current_experiment = None
        self.results_cache = {}
        
    def create_experiment(self,
                         experiment_name: str,
                         experiment_config: Dict) -> str:
        """
        Create new experiment
        """
        # Create experiment directory
        exp_dir = self.results_dir / self.current_version / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
            
        # Create subdirectories
        (exp_dir / "metrics").mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "visualizations").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        self.current_experiment = experiment_name
        
        return str(exp_dir)
    
    def save_metrics(self,
                    metrics: Dict[str, Any],
                    step: Optional[int] = None,
                    tag: str = "train"):
        """
        Save experiment metrics
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        
        # Create filename
        if step is not None:
            filename = f"{tag}_metrics_step_{step}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{tag}_metrics_{timestamp}.json"
            
        # Save metrics
        metrics_file = exp_dir / "metrics" / filename
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
            
        # Update cache
        cache_key = f"{self.current_experiment}_{tag}"
        if cache_key not in self.results_cache:
            self.results_cache[cache_key] = []
        self.results_cache[cache_key].append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    
    def save_model(self,
                  model: Any,
                  model_name: str,
                  metadata: Optional[Dict] = None):
        """
        Save trained model
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        
        # Save model
        model_file = exp_dir / "models" / f"{model_name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metadata
        if metadata:
            meta_file = exp_dir / "models" / f"{model_name}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def save_visualization(self,
                          figure: Any,
                          name: str,
                          format: str = "html"):
        """
        Save visualization
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        
        vis_file = exp_dir / "visualizations" / f"{name}.{format}"
        
        if hasattr(figure, 'savefig'):
            # Matplotlib figure
            figure.savefig(vis_file)
        elif hasattr(figure, 'write_html'):
            # Plotly figure
            figure.write_html(vis_file)
        elif hasattr(figure, 'save'):
            # Other visualization formats
            figure.save(vis_file)
        else:
            # Try to serialize
            with open(vis_file, 'wb') as f:
                pickle.dump(figure, f)
    
    def log_message(self, 
                   message: str,
                   level: str = "INFO"):
        """
        Log experiment message
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        
        log_file = exp_dir / "logs" / "experiment.log"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    
    def get_experiment_summary(self,
                              experiment_name: Optional[str] = None) -> Dict:
        """
        Get experiment summary
        """
        if experiment_name is None:
            if self.current_experiment is None:
                raise ValueError("No experiment specified")
            experiment_name = self.current_experiment
            
        exp_dir = self.results_dir / self.current_version / experiment_name
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")
            
        # Load config
        config_file = exp_dir / "config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Get metrics files
        metrics_dir = exp_dir / "metrics"
        metrics_files = list(metrics_dir.glob("*.json"))
        
        # Get model files
        models_dir = exp_dir / "models"
        model_files = list(models_dir.glob("*.pkl"))
        
        # Get latest metrics
        latest_metrics = {}
        if metrics_files:
            latest_file = max(metrics_files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                latest_metrics = json.load(f)
                
        summary = {
            'experiment_name': experiment_name,
            'version': self.current_version,
            'config': config,
            'num_metrics_files': len(metrics_files),
            'num_models': len(model_files),
            'latest_metrics': latest_metrics,
            'directory': str(exp_dir),
            'created_at': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat()
        }
        
        return summary
    
    def compare_experiments(self,
                          experiment_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments
        """
        comparisons = []
        
        for exp_name in experiment_names:
            try:
                summary = self.get_experiment_summary(exp_name)
                
                comparison = {
                    'experiment': exp_name,
                    'config_hash': hashlib.md5(
                        json.dumps(summary['config'], sort_keys=True).encode()
                    ).hexdigest()[:8]
                }
                
                # Extract key metrics
                if 'latest_metrics' in summary:
                    for key, value in summary['latest_metrics'].items():
                        if isinstance(value, (int, float)):
                            comparison[key] = value
                            
                comparisons.append(comparison)
                
            except Exception as e:
                print(f"Error processing experiment {exp_name}: {e}")
                
        return pd.DataFrame(comparisons)
    
    def export_results(self,
                      export_format: str = "json",
                      include_models: bool = False) -> str:
        """
        Export experiment results
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment")
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        export_dir = exp_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"export_{timestamp}.{export_format}"
        
        # Collect all data
        export_data = {
            'experiment': self.current_experiment,
            'version': self.current_version,
            'export_time': timestamp,
            'summary': self.get_experiment_summary(),
            'all_metrics': self._collect_all_metrics()
        }
        
        if export_format == "json":
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif export_format == "csv":
            # Convert metrics to CSV
            metrics_df = self._metrics_to_dataframe()
            metrics_df.to_csv(export_file.with_suffix('.csv'), index=False)
        elif export_format == "parquet":
            metrics_df = self._metrics_to_dataframe()
            metrics_df.to_parquet(export_file.with_suffix('.parquet'))
            
        return str(export_file)
    
    def _collect_all_metrics(self) -> List[Dict]:
        """Collect all metrics from experiment"""
        if self.current_experiment is None:
            return []
            
        exp_dir = self.results_dir / self.current_version / self.current_experiment
        metrics_dir = exp_dir / "metrics"
        
        all_metrics = []
        for metrics_file in metrics_dir.glob("*.json"):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append({
                    'file': metrics_file.name,
                    'metrics': metrics
                })
                
        return all_metrics
    
    def _metrics_to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame"""
        all_metrics = self._collect_all_metrics()
        
        rows = []
        for item in all_metrics:
            row = {'file': item['file']}
            
            # Flatten metrics
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    if isinstance(value, dict):
                        flatten_dict(value, prefix + key + '_')
                    elif isinstance(value, (int, float, str)):
                        row[prefix + key] = value
                        
            flatten_dict(item['metrics'])
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    @staticmethod
    def _generate_version() -> str:
        """Generate version string"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"