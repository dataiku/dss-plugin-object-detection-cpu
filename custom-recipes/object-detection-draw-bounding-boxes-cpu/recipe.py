# -*- coding: utf-8 -*-
import shutil

import dataiku
import os.path
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
import misc_utils

src_folder = dataiku.Folder(get_input_names_for_role('images')[0])
dst_folder = dataiku.Folder(get_output_names_for_role('output')[0])

configs = get_recipe_config()

label_caption = configs.get('draw_label', False)
confidence_caption = configs.get('draw_confidence', False)

bboxes = dataiku.Dataset(get_input_names_for_role('bbox')[0]).get_dataframe()

paths = bboxes.path.unique().tolist()

ids = bboxes.class_name.unique().tolist()

for path in paths:
    df = bboxes[bboxes.path == path]
    
    src_path = os.path.join(src_folder.get_path(), path)
    dst_path = os.path.join(dst_folder.get_path(), path)

    if len(df) == 0:
        shutil.copy(src_path, dst_path)
        continue
    
    misc_utils.draw_bboxes(src_path, src_folder, dst_path, dst_folder, df, label_caption, confidence_caption, ids)
   