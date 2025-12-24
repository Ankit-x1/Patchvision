import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import threading
import queue

class RealTimeDashboard:
    """
    Real-time monitoring dashboard for industrial vision
    """
    
    def __init__(self, 
                 title: str = "PatchVision Dashboard",
                 update_interval: int = 1000):
        self.title = title
        self.update_interval = update_interval
        self.data_queue = queue.Queue()
        self.metrics = {}
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY]
        )
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, 
                           className="text-center mb-4",
                           style={'color': '#00ff9d'})
                ])
            ]),
            
            # Main metrics row
            dbc.Row([
                dbc.Col(self._create_metric_card(
                    "Throughput", "images/sec", "throughput"
                ), width=3),
                dbc.Col(self._create_metric_card(
                    "Latency", "ms", "latency"
                ), width=3),
                dbc.Col(self._create_metric_card(
                    "Accuracy", "%", "accuracy"
                ), width=3),
                dbc.Col(self._create_metric_card(
                    "GPU Usage", "%", "gpu_usage"
                ), width=3)
            ], className="mb-4"),
            
            # Charts row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='throughput-chart')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='latency-chart')
                ], width=6)
            ], className="mb-4"),
            
            # 3D Visualization
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='3d-visualization')
                ], width=12)
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Start", id="btn-start", color="success"),
                        dbc.Button("Pause", id="btn-pause", color="warning"),
                        dbc.Button("Reset", id="btn-reset", color="danger")
                    ]),
                    dcc.Interval(
                        id='interval-update',
                        interval=self.update_interval,
                        n_intervals=0
                    )
                ], width=12, className="mt-4")
            ])
        ], fluid=True)
    
    def _create_metric_card(self, title: str, unit: str, id: str):
        """Create metric card component"""
        return dbc.Card([
            dbc.CardBody([
                html.H4(title, className="card-title"),
                html.H2("--", id=f"{id}-value", className="card-text"),
                html.P(unit, className="card-text text-muted")
            ])
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('throughput-value', 'children'),
             Output('latency-value', 'children'),
             Output('accuracy-value', 'children'),
             Output('gpu-usage-value', 'children')],
            [Input('interval-update', 'n_intervals')]
        )
        def update_metrics(n):
            """Update metric values"""
            return [
                f"{self.metrics.get('throughput', 0):.1f}",
                f"{self.metrics.get('latency', 0):.1f}",
                f"{self.metrics.get('accuracy', 0):.1f}",
                f"{self.metrics.get('gpu_usage', 0):.1f}"
            ]
        
        @self.app.callback(
            Output('throughput-chart', 'figure'),
            [Input('interval-update', 'n_intervals')]
        )
        def update_throughput_chart(n):
            """Update throughput chart"""
            fig = go.Figure()
            
            # Add real-time data
            if 'throughput_history' in self.metrics:
                fig.add_trace(go.Scatter(
                    y=self.metrics['throughput_history'],
                    mode='lines',
                    name='Throughput',
                    line=dict(color='#00ff9d', width=2)
                ))
                
            fig.update_layout(
                title="Throughput Over Time",
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title="Images/sec"
            )
            
            return fig
        
        @self.app.callback(
            Output('3d-visualization', 'figure'),
            [Input('interval-update', 'n_intervals')]
        )
        def update_3d_visualization(n):
            """Update 3D visualization"""
            fig = go.Figure()
            
            # Sample 3D data
            if 'point_cloud' in self.metrics:
                points = self.metrics['point_cloud']
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='#0088ff')
                ))
                
            fig.update_layout(
                title="Feature Point Cloud",
                template="plotly_dark",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                )
            )
            
            return fig
    
    def update_data(self, data: Dict):
        """Update dashboard data"""
        self.data_queue.put(data)
        
    def start_server(self, host: str = "0.0.0.0", port: int = 8050):
        """Start dashboard server"""
        threading.Thread(
            target=self.app.run_server,
            kwargs={'host': host, 'port': port, 'debug': False},
            daemon=True
        ).start()
        print(f"Dashboard running at http://{host}:{port}")

class PerformanceMonitor:
    """
    Performance monitoring and alerting
    """
    
    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or {
            'latency': 100,  # ms
            'throughput': 30,  # images/sec
            'accuracy': 0.95,  # 95%
            'memory': 0.9  # 90%
        }
        
        self.alerts = []
        self.history = []
        
    def monitor(self, metrics: Dict):
        """
        Monitor metrics and generate alerts
        """
        self.history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        # Check thresholds
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                
                if metric == 'accuracy' and value < threshold:
                    self._add_alert(f"Low accuracy: {value:.2f} < {threshold}")
                elif metric == 'latency' and value > threshold:
                    self._add_alert(f"High latency: {value:.1f}ms > {threshold}ms")
                elif metric == 'throughput' and value < threshold:
                    self._add_alert(f"Low throughput: {value:.1f} < {threshold}")
                elif metric == 'memory' and value > threshold:
                    self._add_alert(f"High memory usage: {value:.1%} > {threshold:.0%}")
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        if not self.history:
            return {}
            
        df = pd.DataFrame(self.history)
        
        return {
            'summary': {
                'avg_latency': df['latency'].mean(),
                'avg_throughput': df['throughput'].mean(),
                'avg_accuracy': df['accuracy'].mean(),
                'total_alerts': len(self.alerts)
            },
            'trends': {
                'latency_trend': self._compute_trend(df['latency']),
                'throughput_trend': self._compute_trend(df['throughput'])
            },
            'alerts': self.alerts[-10:]  # Last 10 alerts
        }
    
    def _add_alert(self, message: str):
        """Add alert to history"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': 'warning'
        }
        self.alerts.append(alert)
        print(f"ALERT: {message}")
    
    @staticmethod
    def _compute_trend(series: pd.Series) -> str:
        """Compute trend direction"""
        if len(series) < 2:
            return "stable"
            
        recent = series.iloc[-5:].mean() if len(series) >= 5 else series.iloc[-1]
        older = series.iloc[-10:-5].mean() if len(series) >= 10 else series.iloc[0]
        
        if recent > older * 1.1:
            return "increasing"
        elif recent < older * 0.9:
            return "decreasing"
        else:
            return "stable"