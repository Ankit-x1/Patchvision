"""
Dashboard Manager for PatchVision
Real-time monitoring and visualization dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Any, Optional, Callable
import time
import threading
from collections import deque
import json


class DashboardManager:
    """Real-time dashboard for monitoring PatchVision operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboard_config = config.get("vision", {}).get("dashboard", {})
        self.update_interval = (
            self.dashboard_config.get("update_interval", 1000) / 1000.0
        )
        self.port = self.dashboard_config.get("port", 8050)

        # Data storage for real-time plotting
        self.max_points = 100
        self.metrics_history = {
            "inference_time": deque(maxlen=self.max_points),
            "fps": deque(maxlen=self.max_points),
            "memory_usage": deque(maxlen=self.max_points),
            "cpu_usage": deque(maxlen=self.max_points),
            "detection_count": deque(maxlen=self.max_points),
            "confidence_avg": deque(maxlen=self.max_points),
        }

        # Plots
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.animation = None
        self.is_running = False

        # Callbacks
        self.data_callbacks = []

    def initialize_dashboard(self):
        """Initialize the dashboard layout"""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor("#0a0a0a")

        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Performance metrics
        self.axes["inference_time"] = self.fig.add_subplot(gs[0, 0])
        self.axes["fps"] = self.fig.add_subplot(gs[0, 1])
        self.axes["memory"] = self.fig.add_subplot(gs[0, 2])

        # Detection metrics
        self.axes["detections"] = self.fig.add_subplot(gs[1, 0])
        self.axes["confidence"] = self.fig.add_subplot(gs[1, 1])
        self.axes["accuracy"] = self.fig.add_subplot(gs[1, 2])

        # System info and status
        self.axes["system"] = self.fig.add_subplot(gs[2, :2])
        self.axes["status"] = self.fig.add_subplot(gs[2, 2])

        # Initialize lines for each plot
        for metric in self.metrics_history:
            if metric in self.axes:
                (line,) = self.axes[metric].plot([], [], "g-", linewidth=2)
                self.lines[metric] = line
                self._setup_metric_plot(metric)

        # Setup static plots
        self._setup_system_info()
        self._setup_status_panel()

        self.fig.suptitle("PatchVision Dashboard", fontsize=20, color="white", y=0.98)

    def _setup_metric_plot(self, metric: str):
        """Setup individual metric plot"""
        ax = self.axes[metric]
        ax.set_facecolor("#1a1a1a")
        ax.set_title(metric.replace("_", " ").title(), color="white", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set labels and limits
        if metric == "inference_time":
            ax.set_ylabel("Time (ms)", color="white")
            ax.set_ylim(0, 200)
        elif metric == "fps":
            ax.set_ylabel("FPS", color="white")
            ax.set_ylim(0, 60)
        elif metric == "memory_usage":
            ax.set_ylabel("Memory (MB)", color="white")
            ax.set_ylim(0, 8000)
        elif metric == "detection_count":
            ax.set_ylabel("Count", color="white")
            ax.set_ylim(0, 100)
        elif metric == "confidence_avg":
            ax.set_ylabel("Confidence", color="white")
            ax.set_ylim(0, 1)

        ax.set_xlabel("Time", color="white")
        ax.tick_params(colors="white")

    def _setup_system_info(self):
        """Setup system information panel"""
        ax = self.axes["system"]
        ax.set_facecolor("#1a1a1a")
        ax.set_title("System Information", color="white", fontsize=12)
        ax.axis("off")

        # Display system info text
        info_text = """
        CPU: {cpu_info}
        GPU: {gpu_info}
        Memory: {memory_info}
        Device: {device}
        Model: {model}
        Config: {config}
        """.format(
            cpu_info="Intel i7-9700K",
            gpu_info="NVIDIA RTX 3080",
            memory_info="32GB DDR4",
            device="CUDA",
            model="PatchVision-v1.0",
            config="Industrial",
        )

        ax.text(
            0.05,
            0.95,
            info_text,
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def _setup_status_panel(self):
        """Setup status panel"""
        ax = self.axes["status"]
        ax.set_facecolor("#1a1a1a")
        ax.set_title("System Status", color="white", fontsize=12)
        ax.axis("off")

        status_text = """
        ● Status: RUNNING
        ● Uptime: 00:00:00
        ● Frames Processed: 0
        ● Errors: 0
        ● Warnings: 0
        """

        self.status_text_obj = ax.text(
            0.05,
            0.95,
            status_text,
            transform=ax.transAxes,
            color="#00ff41",
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics with new values"""
        timestamp = time.time()

        for metric, value in metrics.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append((timestamp, value))

    def add_data_callback(self, callback: Callable[[], Dict[str, float]]):
        """Add callback for real-time data collection"""
        self.data_callbacks.append(callback)

    def _collect_data(self):
        """Collect data from all callbacks"""
        for callback in self.data_callbacks:
            try:
                data = callback()
                self.update_metrics(data)
            except Exception as e:
                print(f"Dashboard callback error: {e}")

    def _update_plot(self, frame):
        """Update plot animation"""
        if not self.is_running:
            return []

        # Collect new data
        self._collect_data()

        updated_lines = []

        # Update each metric plot
        for metric in self.metrics_history:
            if metric in self.lines and len(self.metrics_history[metric]) > 0:
                data = list(self.metrics_history[metric])
                if data:
                    times, values = zip(*data)
                    # Normalize times to start from 0
                    if len(times) > 1:
                        times = [(t - times[0]) for t in times]

                    self.lines[metric].set_data(times, values)

                    # Update x-axis limits
                    ax = self.axes[metric]
                    if len(times) > 1:
                        ax.set_xlim(0, max(times))
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)

                updated_lines.append(self.lines[metric])

        # Update status panel
        self._update_status()

        return updated_lines

    def _update_status(self):
        """Update status panel"""
        # Calculate uptime
        if hasattr(self, "start_time"):
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            uptime_str = "00:00:00"

        # Count processed frames
        total_frames = len(self.metrics_history.get("inference_time", []))

        status_text = f"""
        ● Status: {"RUNNING" if self.is_running else "STOPPED"}
        ● Uptime: {uptime_str}
        ● Frames Processed: {total_frames}
        ● Errors: 0
        ● Warnings: 0
        """

        if hasattr(self, "status_text_obj"):
            self.status_text_obj.set_text(status_text)

    def start(self):
        """Start the dashboard"""
        if self.fig is None:
            self.initialize_dashboard()

        self.is_running = True
        self.start_time = time.time()

        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            interval=int(self.update_interval * 1000),
            blit=True,
            cache_frame_data=False,
        )

        plt.show(block=False)
        print(f"Dashboard started on port {self.port}")

    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        if self.animation is not None:
            self.animation.event_source.stop()
        print("Dashboard stopped")

    def save_dashboard_state(self, filename: str):
        """Save current dashboard state to file"""
        state = {
            "timestamp": time.time(),
            "metrics_history": {k: list(v) for k, v in self.metrics_history.items()},
            "is_running": self.is_running,
        }

        with open(filename, "w") as f:
            json.dump(state, f, indent=2)

    def load_dashboard_state(self, filename: str):
        """Load dashboard state from file"""
        try:
            with open(filename, "r") as f:
                state = json.load(f)

            for metric, data in state.get("metrics_history", {}).items():
                if metric in self.metrics_history:
                    self.metrics_history[metric].clear()
                    self.metrics_history[metric].extend(data)

        except Exception as e:
            print(f"Error loading dashboard state: {e}")

    def close(self):
        """Close the dashboard"""
        self.stop()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
