import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np

# Mock dependencies before importing misc_utils
sys.modules['cv2'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['keras.optimizers'] = MagicMock()
sys.modules['keras.callbacks'] = MagicMock()
sys.modules['keras.utils'] = MagicMock()
sys.modules['keras.models'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['keras_retinanet'] = MagicMock()
sys.modules['keras_retinanet.models'] = MagicMock()
sys.modules['keras_retinanet.models.resnet'] = MagicMock()
sys.modules['keras_retinanet.models.retinanet'] = MagicMock()
sys.modules['keras_retinanet.utils'] = MagicMock()
sys.modules['keras_retinanet.utils.model'] = MagicMock()
sys.modules['keras_retinanet.utils.image'] = MagicMock()
sys.modules['keras_retinanet.utils.visualization'] = MagicMock()
sys.modules['keras_retinanet.utils.colors'] = MagicMock()

# Add python-lib to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-lib')))

import misc_utils

class TestMiscUtils(unittest.TestCase):

    def test_mkv_to_mp4(self):
        def isfile_side_effect(path):
            # test.mkv exists, test.mp4 exists only after conversion
            return path in ('test.mkv', 'test.mp4')

        with patch('subprocess.call') as mock_call, \
             patch('os.path.isfile', side_effect=isfile_side_effect) as mock_isfile, \
             patch('os.remove') as mock_remove:

            # Test case 1: Basic conversion
            misc_utils.mkv_to_mp4('test.mkv', remove_mkv=False, has_audio=True, quiet=True)
            mock_call.assert_called()
            args, _ = mock_call.call_args
            # The exact string might vary depending on implementation details, check key parts
            self.assertIn('ffmpeg -i test.mkv', args[0])
            self.assertIn('test.mp4', args[0])

            # Test case 2: Remove MKV after conversion
            misc_utils.mkv_to_mp4('test.mkv', remove_mkv=True, has_audio=False, quiet=False)
            mock_remove.assert_called_with('test.mkv')

    def test_split_dataset(self):
        df = pd.DataFrame({'path': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 'label': range(10)})
        train, val = misc_utils.split_dataset(df, val_split=0.8, shuffle=False)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        
        # Test shuffle
        train_s, val_s = misc_utils.split_dataset(df, val_split=0.8, shuffle=True, seed=42)
        self.assertEqual(len(train_s), 8)
        self.assertEqual(len(val_s), 2)

    def test_get_cm(self):
        unique_vals = ['cat', 'dog', 'bird']
        mapping = misc_utils.get_cm(unique_vals)
        self.assertEqual(mapping, {'cat': 0, 'dog': 1, 'bird': 2})

    def test_jaccard(self):
        box_a = [0, 0, 10, 10]
        box_b = [0, 0, 10, 10] # Identical
        self.assertEqual(misc_utils.jaccard(box_a, box_b), 1.0)
        
        box_c = [10, 10, 20, 20] # No overlap
        self.assertEqual(misc_utils.jaccard(box_a, box_c), 0.0)

        box_d = [5, 5, 15, 15] # Partial overlap
        # Intersection: 5x5=25. Union: 100+100-25=175. IOU: 25/175 = 1/7
        self.assertAlmostEqual(misc_utils.jaccard(box_a, box_d), 1/7)

    def test_compute_metrics(self):
        tp, fp, fn = 10, 0, 0
        p, r, f1 = misc_utils.compute_metrics(tp, fp, fn)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f1, 1.0)

    def test_compute_metrics_zero_precision_recall(self):
        tp, fp, fn = 0, 10, 10
        p, r, f1 = misc_utils.compute_metrics(tp, fp, fn)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    @patch('misc_utils.cv2.putText')
    def test_draw_caption(self, mock_put_text):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        box = [10, 10, 50, 50]
        caption = "test"
        
        misc_utils.draw_caption(image, box, caption)
        self.assertEqual(mock_put_text.call_count, 2) # Called twice for shadow effect
