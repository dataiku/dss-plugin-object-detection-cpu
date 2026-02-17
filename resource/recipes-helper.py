import os
import glob

import pandas as pd
import numpy as np
from PIL import Image

import dataiku
import gpu_utils
import misc_utils
import constants

DEFAULT_MIN_SIDE = 800
DEFAULT_MAX_SIDE = 1333


def get_dataset_info(inputs):
    label_dataset_full_name = get_input_name_from_role(inputs, 'bounding_boxes')
    label_dataset = dataiku.Dataset(label_dataset_full_name)
    
    columns = [c['name'] for c in label_dataset.read_schema()]
    return {
        'columns': columns,
        'can_use_gpu': gpu_utils.can_use_gpu()
    }


def has_confidence(inputs):
    label_dataset_full_name = get_input_name_from_role(inputs, 'bbox')
    label_dataset = dataiku.Dataset(label_dataset_full_name)
    
    for c in label_dataset.read_schema():
        name = c['name']
        if name == 'confidence': return True
    return False


def get_input_name_from_role(inputs, role):
    return [inp for inp in inputs if inp["role"] == role][0]['fullName']

 
def do(payload, config, plugin_config, inputs):
    if 'method' not in payload:
        return {}

    client = dataiku.api_client()

    if payload['method'] == 'get-dataset-info':
        response = get_dataset_info(inputs)
        response.update(get_avg_side(inputs))
        return response
    if payload['method'] == 'get-gpu-info':
        return {'can_use_gpu': gpu_utils.can_use_gpu()}
    if payload['method'] == 'get-confidence':
        return {'has_confidence': has_confidence(inputs)}
    if payload['method'] == 'get-video-info':
        return {
            'can_use_gpu': gpu_utils.can_use_gpu(),
            'columns': get_available_videos(inputs)
        }

    return {}


def get_available_videos(inputs):
    name = get_input_name_from_role(inputs, 'video')
    folder = dataiku.Folder(name)
    
    return [f[1:] for f in folder.list_paths_in_partition()]


def get_avg_side(inputs, n_first=3000):
    """Min side is first quartile, max side is third quartile."""
    image_folder_full_name = get_input_name_from_role(inputs, 'images')
    image_folder = dataiku.Folder(image_folder_full_name)
    folder_path = image_folder.get_path()
    
    paths = image_folder.list_paths_in_partition()[:n_first]
    sides = []
    for path in paths:
        path = os.path.join(folder_path, path[1:])
        try:
            # PIL does not load raster data at this point, so reading size is fast.
            with Image.open(path) as img:
                w, h = img.size
            sides.append(w)
            sides.append(h)
        except Exception:
            # Skip unreadable files during setup stat computation.
            continue

    if len(sides) == 0:
        return {
            'min_side': DEFAULT_MIN_SIDE,
            'max_side': DEFAULT_MAX_SIDE
        }

    sides = np.array(sides)
    
    return {
        'min_side': int(np.percentile(sides, 25)),
        'max_side': int(np.percentile(sides, 75))
    }
