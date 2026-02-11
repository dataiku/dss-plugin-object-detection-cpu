import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import json

# Mock dependencies
sys.modules['keras_retinanet'] = MagicMock()
sys.modules['keras_retinanet.preprocessing'] = MagicMock()
sys.modules['keras_retinanet.preprocessing.csv_generator'] = MagicMock()

# Define a mock Generator class that DfGenerator can inherit from
class MockGenerator:
    def __init__(self, **kwargs):
        pass

sys.modules['keras_retinanet.preprocessing.csv_generator'].Generator = MockGenerator
sys.modules['keras_retinanet.preprocessing.csv_generator'].CSVGenerator = MockGenerator

# Add python-lib to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python-lib')))

from dfgenerator import DfGenerator

class TestDfGenerator(unittest.TestCase):

    def setUp(self):
        self.cols = {
            'col_filename': 'path',
            'col_label': 'label',
            'col_x1': 'x1', 'col_y1': 'y1',
            'col_x2': 'x2', 'col_y2': 'y2',
            'single_column_data': False
        }
        self.class_mapping = {'cat': 0}

    def test_init_standard_format(self):
        df = pd.DataFrame({
            'path': ['img1.jpg'],
            'label': ['cat'],
            'x1': [10], 'y1': [10], 'x2': [100], 'y2': [100]
        })
        
        gen = DfGenerator(df, self.class_mapping, self.cols)
        
        self.assertIn('img1.jpg', gen.image_data)
        self.assertEqual(len(gen.image_data['img1.jpg']), 1)
        self.assertEqual(gen.image_data['img1.jpg'][0]['class'], 'cat')

    def test_init_json_format(self):
        self.cols['single_column_data'] = True
        label_json = json.dumps([
            {"top": 10, "left": 10, "width": 90, "height": 90, "label": "cat"}
        ])
        df = pd.DataFrame({
            'path': ['img2.jpg'],
            'label': [label_json]
        })
        
        gen = DfGenerator(df, self.class_mapping, self.cols)
        
        self.assertIn('img2.jpg', gen.image_data)
        data = gen.image_data['img2.jpg'][0]
        self.assertEqual(data['class'], 'cat')
        self.assertEqual(data['x1'], 10)
        self.assertEqual(data['y1'], 10)
        self.assertEqual(data['x2'], 100) # 10+90
        self.assertEqual(data['y2'], 100) # 10+90

    def test_len(self):
        df = pd.DataFrame({
            'path': ['img1.jpg', 'img2.jpg'],
            'label': ['cat', 'cat'],
            'x1': [0,0], 'y1': [0,0], 'x2': [1,1], 'y2': [1,1]
        })
        gen = DfGenerator(df, self.class_mapping, self.cols)
        self.assertEqual(len(gen), 2)
