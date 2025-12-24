import unittest
import numpy as np
from patchvision.deploy.api import APIServer
from patchvision.deploy.edge import EdgeDeployer
from patchvision.deploy.monitoring import ProductionMonitor

class TestAPIServer(unittest.TestCase):
    
    def test_server_initialization(self):
        server = APIServer(host="127.0.0.1", port=8000)
        self.assertIsNotNone(server.app)
        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 8000)

class TestEdgeDeployer(unittest.TestCase):
    
    def test_deployer_initialization(self):
        deployer = EdgeDeployer(target_device="raspberry_pi")
        self.assertEqual(deployer.target_device, "raspberry_pi")
        self.assertEqual(deployer.optimization_level, "high")

class TestProductionMonitor(unittest.TestCase):
    
    def setUp(self):
        self.monitor = ProductionMonitor(monitor_interval=1)
    
    def test_monitor_initialization(self):
        self.assertEqual(self.monitor.monitor_interval, 1)
        self.assertEqual(self.monitor.history_size, 1000)
    
    def test_collect_metrics(self):
        metrics = self.monitor._collect_metrics()
        self.assertIn('timestamp', metrics)
        self.assertIn('system', metrics)
        self.assertIn('application', metrics)

if __name__ == '__main__':
    unittest.main()