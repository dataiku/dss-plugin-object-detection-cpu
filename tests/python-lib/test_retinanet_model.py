import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['keras.optimizers'] = MagicMock()
sys.modules['keras.callbacks'] = MagicMock()
sys.modules['keras.utils'] = MagicMock()
sys.modules['keras.models'] = MagicMock()
sys.modules['keras_retinanet'] = MagicMock()
sys.modules['keras_retinanet.models'] = MagicMock()
sys.modules['keras_retinanet.models.resnet'] = MagicMock()
sys.modules['keras_retinanet.models.retinanet'] = MagicMock()
sys.modules['keras_retinanet.utils'] = MagicMock()
sys.modules['keras_retinanet.utils.model'] = MagicMock()
sys.modules['keras_retinanet.utils.image'] = MagicMock()
sys.modules['keras_retinanet.utils.transform'] = MagicMock()

# Add python-lib to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-lib')))

import retinanet_model

class TestRetinaNetModel(unittest.TestCase):

    @patch('retinanet_model.resnet50_retinanet')
    @patch('retinanet_model.multi_gpu_model')
    @patch('retinanet_model.tf.device')
    def test_get_model(self, mock_device, mock_multi_gpu, mock_resnet):
        mock_model = MagicMock()
        mock_resnet.return_value = mock_model

        # Case 1: CPU (n_gpu=None)
        retinanet_model.get_model('weights.h5', 10, freeze=False, n_gpu=None)
        mock_resnet.assert_called()
        mock_multi_gpu.assert_not_called()
        mock_model.load_weights.assert_called_with('weights.h5', by_name=True, skip_mismatch=True)

        # Case 2: Multi-GPU
        mock_model.load_weights.reset_mock()
        retinanet_model.get_model('weights.h5', 10, freeze=False, n_gpu=2)
        mock_multi_gpu.assert_called()
        mock_model.load_weights.assert_called_with('weights.h5', by_name=True, skip_mismatch=True)
        
    @patch('retinanet_model.get_model')
    @patch('retinanet_model.retinanet_bbox')
    def test_get_test_model(self, mock_bbox, mock_get_model):
        mock_model = MagicMock()
        mock_get_model.return_value = (mock_model, mock_model)
        
        retinanet_model.get_test_model('weights.h5', 10)
        
        mock_get_model.assert_called_with('weights.h5', 10, freeze=True, n_gpu=1)
        mock_bbox.assert_called_with(model=mock_model)

    def test_compile_model(self):
        model = MagicMock()
        configs = {'optimizer': 'adam', 'lr': 0.001}
        retinanet_model.compile_model(model, configs)
        model.compile.assert_called()

    @patch('retinanet_model.read_image_bgr')
    @patch('retinanet_model.preprocess_image')
    @patch('retinanet_model.resize_image')
    def test_find_objects(self, mock_resize, mock_preprocess, mock_read):
        model = MagicMock()
        # Mock prediction output: boxes, scores, labels
        # Output shape is usually (batch, num_boxes, coords/score/label)
        model.predict_on_batch.return_value = (np.zeros((1, 300, 4)), np.zeros((1, 300)), np.zeros((1, 300)))
        
        mock_read.return_value = np.zeros((100, 100, 3))
        mock_preprocess.return_value = np.zeros((100, 100, 3))
        mock_resize.return_value = (np.zeros((100, 100, 3)), 1.0)
        
        paths = ['img1.jpg']
        boxes, scores, labels = retinanet_model.find_objects(model, paths)
        
        self.assertEqual(len(boxes), 1)
        mock_read.assert_called_with('img1.jpg')

    @patch('retinanet_model.random_transform_generator')
    def test_get_random_augmentator(self, mock_gen):
        configs = {
            'min_rotation': 0, 'max_rotation': 1,
            'min_trans': 0, 'max_trans': 1,
            'min_shear': 0, 'max_shear': 1,
            'min_scaling': 0, 'max_scaling': 1,
            'flip_x': 0.5, 'flip_y': 0.5
        }
        retinanet_model.get_random_augmentator(configs)
        mock_gen.assert_called_with(
            min_rotation=0.0,
            max_rotation=1.0,
            min_translation=(0.0, 0.0),
            max_translation=(1.0, 1.0),
            min_shear=0.0,
            max_shear=1.0,
            min_scaling=(0.0, 0.0),
            max_scaling=(1.0, 1.0),
            flip_x_chance=0.5,
            flip_y_chance=0.5
        )
