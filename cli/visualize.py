import time

def run_visualization(args, config):
    """Run the visualization dashboard."""
    print("Starting visualization mode...")
    from patchvision.vision.dashboard import DashboardManager
    
    dashboard = DashboardManager(config)
    
    def collect_metrics():
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().used / (1024 * 1024),  # MB
            'inference_time': 0.0,
            'fps': 0.0,
            'detection_count': 0,
            'confidence_avg': 0.0
        }
    
    dashboard.add_data_callback(collect_metrics)
    dashboard.start()
    
    print("Dashboard running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping dashboard...")
        dashboard.stop()
        dashboard.close()
