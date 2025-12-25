"""
Model state management for PatchVision
"""

import json
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import numpy as np


class ModelState(Enum):
    """Model operational states"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SAVING = "saving"
    UPDATING = "updating"


@dataclass
class StateSnapshot:
    """Model state snapshot"""
    timestamp: float
    state: ModelState
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class ModelStateManager:
    """
    Advanced model state management with snapshots and transitions
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.current_state = ModelState.UNINITIALIZED
        self.state_history: List[StateSnapshot] = []
        self.state_lock = threading.Lock()
        
        # State transition rules
        self.transition_rules = self._define_transition_rules()
        
        # State change callbacks
        self.state_callbacks: List[Callable[[ModelState, ModelState], None]] = []
        
        # Performance metrics
        self.metrics: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Auto-save configuration
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5 minutes
        self.max_snapshots = 50
        
        # Background auto-save thread
        self._stop_auto_save = threading.Event()
        self._auto_save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self._auto_save_thread.start()
    
    def _define_transition_rules(self) -> Dict[ModelState, List[ModelState]]:
        """Define valid state transitions"""
        return {
            ModelState.UNINITIALIZED: [ModelState.LOADING],
            ModelState.LOADING: [ModelState.READY, ModelState.ERROR],
            ModelState.READY: [ModelState.RUNNING, ModelState.SAVING, ModelState.UPDATING, ModelState.ERROR],
            ModelState.RUNNING: [ModelState.READY, ModelState.ERROR, ModelState.SAVING],
            ModelState.ERROR: [ModelState.READY, ModelState.LOADING],
            ModelState.SAVING: [ModelState.READY, ModelState.ERROR],
            ModelState.UPDATING: [ModelState.READY, ModelState.ERROR, ModelState.LOADING]
        }
    
    def transition_to(self, new_state: ModelState, error_message: Optional[str] = None) -> bool:
        """
        Transition to a new state
        
        Args:
            new_state: Target state
            error_message: Error message if transitioning to ERROR state
            
        Returns:
            True if transition successful, False otherwise
        """
        with self.state_lock:
            # Check if transition is valid
            if new_state not in self.transition_rules.get(self.current_state, []):
                print(f"Invalid state transition: {self.current_state} -> {new_state}")
                return False
            
            old_state = self.current_state
            self.current_state = new_state
            
            # Create state snapshot
            snapshot = StateSnapshot(
                timestamp=time.time(),
                state=new_state,
                metrics=self.metrics.copy(),
                metadata=self.metadata.copy(),
                error_message=error_message
            )
            
            self.state_history.append(snapshot)
            
            # Limit history size
            if len(self.state_history) > self.max_snapshots:
                self.state_history = self.state_history[-self.max_snapshots:]
            
            # Notify callbacks
            for callback in self.state_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    print(f"State callback error: {e}")
            
            # Auto-save if enabled
            if self.auto_save_enabled:
                self._save_state_snapshot(snapshot)
            
            return True
    
    def get_current_state(self) -> ModelState:
        """Get current model state"""
        with self.state_lock:
            return self.current_state
    
    def get_state_history(self, limit: Optional[int] = None) -> List[StateSnapshot]:
        """Get state history"""
        with self.state_lock:
            if limit:
                return self.state_history[-limit:]
            return self.state_history.copy()
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        with self.state_lock:
            self.metrics.update(metrics)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        with self.state_lock:
            return self.metrics.copy()
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update metadata"""
        with self.state_lock:
            self.metadata.update(metadata)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current metadata"""
        with self.state_lock:
            return self.metadata.copy()
    
    def add_state_callback(self, callback: Callable[[ModelState, ModelState], None]):
        """Add state change callback"""
        self.state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[ModelState, ModelState], None]):
        """Remove state change callback"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
    
    def is_ready(self) -> bool:
        """Check if model is ready for operations"""
        return self.current_state == ModelState.READY
    
    def is_running(self) -> bool:
        """Check if model is currently running"""
        return self.current_state == ModelState.RUNNING
    
    def has_error(self) -> bool:
        """Check if model is in error state"""
        return self.current_state == ModelState.ERROR
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        with self.state_lock:
            for snapshot in reversed(self.state_history):
                if snapshot.state == ModelState.ERROR and snapshot.error_message:
                    return snapshot.error_message
            return None
    
    def reset_state(self):
        """Reset to uninitialized state"""
        self.transition_to(ModelState.UNINITIALIZED)
        with self.state_lock:
            self.metrics.clear()
            self.metadata.clear()
    
    def load_state_snapshot(self, snapshot_path: str) -> bool:
        """Load state from snapshot file"""
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            snapshot = StateSnapshot(
                timestamp=snapshot_data['timestamp'],
                state=ModelState(snapshot_data['state']),
                metrics=snapshot_data['metrics'],
                metadata=snapshot_data['metadata'],
                error_message=snapshot_data.get('error_message')
            )
            
            with self.state_lock:
                self.current_state = snapshot.state
                self.metrics = snapshot.metrics
                self.metadata = snapshot.metadata
                self.state_history.append(snapshot)
            
            return True
        except Exception as e:
            print(f"Failed to load state snapshot: {e}")
            return False
    
    def save_checkpoint(self, checkpoint_name: str, data: Any) -> str:
        """Save model checkpoint with state"""
        checkpoint_dir = self.model_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_{int(time.time())}.pkl"
        
        checkpoint_data = {
            'state': self.current_state.value,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': time.time(),
            'model_data': data
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            return str(checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Any:
        """Load model checkpoint and restore state"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            with self.state_lock:
                self.current_state = ModelState(checkpoint_data['state'])
                self.metrics = checkpoint_data['metrics']
                self.metadata = checkpoint_data['metadata']
            
            return checkpoint_data['model_data']
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        checkpoint_dir = self.model_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        
        return [str(p) for p in checkpoint_dir.glob("*.pkl")]
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """Delete checkpoint"""
        try:
            Path(checkpoint_path).unlink()
            return True
        except:
            return False
    
    def _save_state_snapshot(self, snapshot: StateSnapshot):
        """Save state snapshot to disk"""
        snapshots_dir = self.model_dir / "state_snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        
        snapshot_file = snapshots_dir / f"snapshot_{int(snapshot.timestamp)}.json"
        
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2)
        except Exception as e:
            print(f"Failed to save state snapshot: {e}")
    
    def _auto_save_worker(self):
        """Background worker for auto-saving"""
        while not self._stop_auto_save.wait(self.auto_save_interval):
            if self.auto_save_enabled and self.current_state != ModelState.UNINITIALIZED:
                snapshot = StateSnapshot(
                    timestamp=time.time(),
                    state=self.current_state,
                    metrics=self.metrics.copy(),
                    metadata=self.metadata.copy()
                )
                self._save_state_snapshot(snapshot)
    
    def cleanup(self):
        """Cleanup resources"""
        self._stop_auto_save.set()
        if self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=1.0)
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get state statistics"""
        with self.state_lock:
            if not self.state_history:
                return {}
            
            state_counts = {}
            total_time = 0
            
            for i, snapshot in enumerate(self.state_history):
                state_name = snapshot.state.value
                state_counts[state_name] = state_counts.get(state_name, 0) + 1
                
                if i > 0:
                    total_time += snapshot.timestamp - self.state_history[i-1].timestamp
            
            return {
                'total_transitions': len(self.state_history),
                'state_distribution': state_counts,
                'average_state_duration': total_time / max(len(self.state_history) - 1, 1),
                'current_state': self.current_state.value,
                'last_transition': self.state_history[-1].timestamp if self.state_history else None
            }
