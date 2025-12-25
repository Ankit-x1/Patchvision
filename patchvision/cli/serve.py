def run_server(args, config):
    """Run the API server."""
    print("Starting API server...")
    from patchvision.deploy.api.server import create_app
    
    app = create_app(config)
    
    import uvicorn
    host = config.get('deploy', {}).get('api', {}).get('host', '0.0.0.0')
    port = config.get('deploy', {}).get('api', {}).get('port', 8000)
    
    print(f"Server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
