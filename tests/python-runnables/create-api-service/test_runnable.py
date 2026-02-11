import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules['dataiku'] = MagicMock()
sys.modules['dataiku.runnables'] = MagicMock()

# Add runnable path
runnable_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../python-runnables/create-api-service'))
sys.path.append(runnable_dir)

import runnable

class TestRunnable(unittest.TestCase):

    def test_get_params_create_new(self):
        config = {
            "model_folder_id": "folder1",
            "create_new_service": True,
            "service_id_new": "new_service",
            "endpoint_id": "ep1",
            "confidence": 0.5
        }
        client = MagicMock()
        project = MagicMock()
        
        # Mock folder list
        # list_managed_folders returns list of dicts-like objects or dicts
        project.list_managed_folders.return_value = [{"id": "folder1"}]
        
        # Mock service list (empty so new service is allowed)
        project.list_api_services.return_value = []
        
        params = runnable.get_params(config, client, project)
        
        self.assertEqual(params['service_id'], 'new_service')
        self.assertEqual(params['model_folder_id'], 'folder1')

    def test_get_model_endpoint_settings(self):
        params = {
            "endpoint_id": "ep1",
            "code_env_name": "env1",
            "model_folder_id": "folder1",
            "confidence": 0.5
        }
        
        settings = runnable.get_model_endpoint_settings(params)
        
        self.assertEqual(settings['id'], 'ep1')
        self.assertEqual(settings['inputFolderRefs'][0]['ref'], 'folder1')
        self.assertIn("score < 0.5", settings['code']) # check substitution
