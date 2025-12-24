import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import json

class HolographicVisualizer:
    """
    3D holographic visualization for industrial data
    """
    
    def __init__(self, 
                 theme: str = 'industrial',
                 interactive: bool = True):
        self.theme = theme
        self.interactive = interactive
        self.colors = self._get_theme_colors(theme)
        
    def create_3d_point_cloud(self,
                             points: np.ndarray,
                             values: Optional[np.ndarray] = None,
                             size: int = 5,
                             opacity: float = 0.8) -> go.Figure:
        """
        Create interactive 3D point cloud
        """
        fig = go.Figure()
        
        if values is None:
            color = self.colors['primary']
        else:
            # Color by values
            color = values
            
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                colorscale='Viridis',
                opacity=opacity,
                showscale=True
            ),
            hovertemplate='<b>X</b>: %{x}<br>' +
                         '<b>Y</b>: %{y}<br>' +
                         '<b>Z</b>: %{z}<extra></extra>'
        ))
        
        self._apply_industrial_theme(fig)
        return fig
    
    def create_feature_volume(self,
                             volume: np.ndarray,
                             threshold: float = 0.5,
                             opacity: float = 0.1) -> go.Figure:
        """
        Create 3D volume visualization
        """
        x, y, z = np.mgrid[:volume.shape[0],
                          :volume.shape[1],
                          :volume.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=threshold,
            isomax=volume.max(),
            opacity=opacity,
            surface_count=25,
            colorscale='Hot'
        ))
        
        self._apply_industrial_theme(fig)
        return fig
    
    def create_multi_view(self,
                         views: List[Dict],
                         titles: List[str]) -> go.Figure:
        """
        Create multi-view 3D visualization
        """
        n_rows = int(np.ceil(len(views) / 2))
        fig = make_subplots(
            rows=n_rows, cols=2,
            specs=[[{'type': 'scatter3d'}] * 2] * n_rows,
            subplot_titles=titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for idx, view in enumerate(views):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            if 'points' in view:
                fig.add_trace(
                    go.Scatter3d(**view),
                    row=row, col=col
                )
            elif 'volume' in view:
                fig.add_trace(
                    go.Volume(**view),
                    row=row, col=col
                )
                
        self._apply_industrial_theme(fig)
        return fig
    
    def create_animation(self,
                        frames: List[np.ndarray],
                        interval: int = 100) -> go.Figure:
        """
        Create animated 3D visualization
        """
        fig = go.Figure()
        
        # Create frames
        fig_frames = []
        for i, frame in enumerate(frames):
            fig_frames.append(
                go.Frame(
                    data=[go.Scatter3d(
                        x=frame[:, 0],
                        y=frame[:, 1],
                        z=frame[:, 2],
                        mode='markers'
                    )],
                    name=f"frame_{i}"
                )
            )
            
        fig.frames = fig_frames
        
        # Animation controls
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": interval, 
                                                 "redraw": True},
                                       "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, 
                                                   "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        self._apply_industrial_theme(fig)
        return fig
    
    def _apply_industrial_theme(self, fig: go.Figure):
        """Apply industrial-style theme"""
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis=dict(
                    gridcolor="gray",
                    showbackground=True,
                    backgroundcolor="rgb(20, 20, 20)"
                ),
                yaxis=dict(
                    gridcolor="gray",
                    showbackground=True,
                    backgroundcolor="rgb(20, 20, 20)"
                ),
                zaxis=dict(
                    gridcolor="gray",
                    showbackground=True,
                    backgroundcolor="rgb(20, 20, 20)"
                ),
                aspectmode='data'
            ),
            margin=dict(r=20, l=10, b=10, t=30)
        )
    
    @staticmethod
    def _get_theme_colors(theme: str) -> Dict:
        """Get color scheme for theme"""
        themes = {
            'industrial': {
                'primary': '#00ff9d',
                'secondary': '#0088ff',
                'background': '#1a1a1a',
                'grid': '#333333'
            },
            'medical': {
                'primary': '#ff6b6b',
                'secondary': '#4ecdc4',
                'background': '#ffffff',
                'grid': '#cccccc'
            },
            'thermal': {
                'primary': '#ff0000',
                'secondary': '#ffff00',
                'background': '#000000',
                'grid': '#444444'
            }
        }
        return themes.get(theme, themes['industrial'])

class Interactive3D:
    """
    Interactive 3D visualization with controls
    """
    
    def __init__(self):
        self.figures = {}
        
    def add_controls(self,
                    fig: go.Figure,
                    controls: List[Dict]) -> go.Figure:
        """
        Add interactive controls to figure
        """
        sliders = []
        buttons = []
        
        for control in controls:
            if control['type'] == 'slider':
                sliders.append(self._create_slider(control))
            elif control['type'] == 'button':
                buttons.append(self._create_button(control))
                
        if sliders:
            fig.update_layout(sliders=sliders)
        if buttons:
            fig.update_layout(updatemenus=buttons)
            
        return fig
    
    def embed_in_notebook(self, fig: go.Figure) -> str:
        """Generate HTML for notebook embedding"""
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    @staticmethod
    def _create_slider(control: Dict) -> Dict:
        """Create slider control"""
        return {
            'active': control.get('active', 0),
            'steps': control.get('steps', []),
            'currentvalue': {
                'prefix': control.get('prefix', 'Value: '),
                'visible': True
            }
        }
    
    @staticmethod
    def _create_button(control: Dict) -> Dict:
        """Create button control"""
        return {
            'buttons': control.get('buttons', []),
            'direction': control.get('direction', 'left'),
            'showactive': control.get('showactive', True)
        }