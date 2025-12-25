"""
Holographic 3D Visualization for PatchVision
Real-time 3D rendering of patches and projections
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Optional
import time


class HolographicRenderer:
    """3D holographic visualization for patch-based processing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig = None
        self.ax = None
        self.point_size = (
            config.get("vision", {}).get("holographic", {}).get("point_size", 3)
        )
        self.opacity = (
            config.get("vision", {}).get("holographic", {}).get("opacity", 0.8)
        )
        self.theme = (
            config.get("vision", {}).get("holographic", {}).get("theme", "industrial")
        )

        # Theme colors
        self.themes = {
            "industrial": {
                "background": "#1a1a1a",
                "patches": "#00ff41",
                "tokens": "#ff6b35",
                "attention": "#00b4d8",
                "defects": "#ff006e",
            },
            "medical": {
                "background": "#0a0a0a",
                "patches": "#4cc9f0",
                "tokens": "#f72585",
                "attention": "#7209b7",
                "defects": "#f77f00",
            },
        }

        self.current_theme = self.themes.get(self.theme, self.themes["industrial"])

    def initialize_plot(self):
        """Initialize 3D plot"""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor(self.current_theme["background"])
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(self.current_theme["background"])

        # Set labels
        self.ax.set_xlabel("X", color="white")
        self.ax.set_ylabel("Y", color="white")
        self.ax.set_zlabel("Z", color="white")
        self.ax.set_title("PatchVision Holographic View", color="white", fontsize=16)

        # Grid styling
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

    def visualize_patches(
        self, patches: List[Dict[str, Any]], tokens: Optional[np.ndarray] = None
    ):
        """Visualize patches in 3D space"""
        if self.fig is None:
            self.initialize_plot()

        self.ax.clear()

        # Extract patch coordinates and data
        x_coords = []
        y_coords = []
        z_coords = []
        colors = []
        sizes = []

        for i, patch in enumerate(patches):
            # Use patch center as coordinates
            x = patch["x"] + patch["width"] / 2
            y = patch["y"] + patch["height"] / 2
            z = 0  # Base layer

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

            # Color based on patch confidence or type
            confidence = patch.get("confidence", 0.5)
            if confidence > 0.8:
                colors.append(self.current_theme["defects"])
            elif confidence > 0.5:
                colors.append(self.current_theme["attention"])
            else:
                colors.append(self.current_theme["patches"])

            # Size based on patch area
            area = patch["width"] * patch["height"]
            sizes.append(area / 100)

        # Plot patches
        self.ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=colors,
            s=sizes,
            alpha=self.opacity,
            edgecolors="white",
            linewidths=0.5,
        )

        # Visualize tokens if available
        if tokens is not None and len(tokens.shape) >= 2:
            token_x = np.random.random(tokens.shape[0]) * 100
            token_y = np.random.random(tokens.shape[0]) * 100
            token_z = np.ones(tokens.shape[0]) * 50  # Elevated layer

            self.ax.scatter(
                token_x,
                token_y,
                token_z,
                c=self.current_theme["tokens"],
                s=self.point_size * 2,
                alpha=self.opacity * 0.7,
                marker="^",
            )

        # Set axis limits
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_zlim(0, 100)

        # Add info text
        info_text = f"Patches: {len(patches)}"
        if tokens is not None:
            info_text += f" | Tokens: {tokens.shape[0]}"
        self.ax.text2D(
            0.02,
            0.98,
            info_text,
            transform=self.ax.transAxes,
            color="white",
            fontsize=10,
            verticalalignment="top",
        )

    def visualize_attention(self, attention_weights: np.ndarray, positions: np.ndarray):
        """Visualize attention weights as connections"""
        if self.fig is None:
            self.initialize_plot()

        # Draw attention connections
        for i in range(attention_weights.shape[0]):
            for j in range(attention_weights.shape[1]):
                weight = attention_weights[i, j]
                if weight > 0.1:  # Threshold for visibility
                    x_line = [positions[i, 0], positions[j, 0]]
                    y_line = [positions[i, 1], positions[j, 1]]
                    z_line = [positions[i, 2], positions[j, 2]]

                    self.ax.plot(
                        x_line,
                        y_line,
                        z_line,
                        color=self.current_theme["attention"],
                        alpha=weight * self.opacity,
                        linewidth=weight * 3,
                    )

    def update_realtime(
        self, patches: List[Dict[str, Any]], tokens: Optional[np.ndarray] = None
    ):
        """Update visualization in real-time"""
        self.visualize_patches(patches, tokens)
        plt.pause(0.001)  # Small pause for animation

    def save_frame(self, filename: str):
        """Save current frame to file"""
        if self.fig is not None:
            self.fig.savefig(
                filename,
                dpi=150,
                bbox_inches="tight",
                facecolor=self.current_theme["background"],
            )

    def close(self):
        """Close the visualization"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
