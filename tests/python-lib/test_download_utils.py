import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.handlers'] = MagicMock()

# Add python-lib to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-lib')))

import download_utils

class TestDownloadUtils(unittest.TestCase):

    def test_get_s3_key(self):
        key = download_utils.get_s3_key('resnet50', 'coco')
        self.assertEqual(key, 'pretrained_models/image/object_detection/resnet50_coco_weights.h5')

    def test_get_s3_key_labels(self):
        key = download_utils.get_s3_key_labels('coco')
        self.assertEqual(key, 'pretrained_models/image/coco/labels.json')

    @patch('boto3.resource')
    def test_download_labels(self, mock_resource):
        mock_bucket = MagicMock()
        mock_resource.return_value.Bucket.return_value = mock_bucket
        
        download_utils.download_labels('coco', 'labels.json')
        
        mock_resource.assert_called_with('s3')
        mock_bucket.download_file.assert_called_with(
            'pretrained_models/image/coco/labels.json',
            'labels.json'
        )

    @patch('boto3.resource')
    def test_download_model(self, mock_resource):
        mock_bucket = MagicMock()
        mock_resource.return_value.Bucket.return_value = mock_bucket
        callback = MagicMock()
        
        # Mock get_obj_size to return something for ProgressTracker
        mock_client = mock_resource.return_value.meta.client
        mock_client.get_object.return_value = {'ContentLength': 100}

        download_utils.download_model('resnet50', 'coco', 'weights.h5', callback)
        
        mock_bucket.download_file.assert_called()
        args, kwargs = mock_bucket.download_file.call_args
        self.assertEqual(args[0], 'pretrained_models/image/object_detection/resnet50_coco_weights.h5')
        self.assertEqual(args[1], 'weights.h5')
        self.assertIn('Callback', kwargs)

    @patch('download_utils.get_obj_size')
    def test_progress_tracker(self, mock_get_size):
        mock_get_size.return_value = 100
        callback = MagicMock()
        resource = MagicMock()
        
        tracker = download_utils.ProgressTracker(resource, 'key', callback)
        tracker(10)
        callback.assert_called_with(10) # 10/100 * 100 = 10%
        
        tracker(40)
        callback.assert_called_with(50) # (10+40)/100 * 100 = 50%
