# Unit Testing Plan for Dataiku DSS Object Detection Plugin

This document outlines the plan for introducing unit tests to the `dss-plugin-object-detection-cpu` project. The goal is to cover existing logic with tests without refactoring the code.

## Test Directory Structure

We will create a `tests` directory at the root of the project to house all test files. The structure will mirror the `python-lib` directory for clarity.

```
tests/
├── python-lib/
│   ├── test_misc_utils.py
│   ├── test_download_utils.py
│   ├── test_gpu_utils.py
│   ├── test_dfgenerator.py
│   └── test_retinanet_model.py
└── python-runnables/
    └── create-api-service/
        └── test_runnable.py
```

## Testing Strategy

We will use `pytest` as the test runner and `unittest.mock` (standard library) or `pytest-mock` for isolation. Since the code interacts heavily with external systems (Dataiku API, file system, AWS S3, TensorFlow/Keras, OpenCV), mocking will be essential.

### 1. `python-lib/misc_utils.py`

*   **`mkv_to_mp4`**:
    *   **Test:** Verify the correct `ffmpeg` command is constructed and passed to `subprocess.call`.
    *   **Mocks:** `subprocess.call`, `os.path.isfile`, `os.remove`.
*   **`split_dataset`**:
    *   **Test:** Pass a Pandas DataFrame and verify the returned train/val split ratios and lack of data overlap.
*   **`get_cm`**:
    *   **Test:** Pass a list of unique values and verify the returned dictionary mapping.
*   **`jaccard`**:
    *   **Test:** Compute Intersection over Union (IoU) for known bounding box coordinates.
*   **`compute_metrics`**:
    *   **Test:** Calculate Precision, Recall, and F1 score for known true positives, false positives, and false negatives.
*   **`draw_bboxes`**:
    *   **Test:** Verify `cv2.imwrite` is called. Ensure `draw_box` and `draw_caption` are called for each valid row in the DataFrame.
    *   **Mocks:** `read_image_bgr` (from `keras_retinanet`), `cv2.imwrite`, `draw_box`, `draw_caption` (internal or mocked).
*   **`draw_caption`**:
    *   **Test:** Verify `cv2.putText` is called with expected arguments.
    *   **Mocks:** `cv2.putText`.

### 2. `python-lib/download_utils.py`

*   **`get_s3_key`, `get_s3_key_labels`**:
    *   **Test:** Verify string formatting for different architectures and datasets.
*   **`download_labels`, `download_model`**:
    *   **Test:** Verify `boto3` client is initialized and `download_file` is called with correct bucket and keys.
    *   **Mocks:** `boto3.resource`.
*   **`ProgressTracker`**:
    *   **Test:** Simulate `__call__` with bytes and verify the callback is invoked with the correct percentage.
    *   **Mocks:** `get_obj_size` (mocked to return a fixed size).

### 3. `python-lib/gpu_utils.py`

*   **`load_gpu_options`**:
    *   **Test:** Verify `n_gpu` and `tf.ConfigProto` settings based on inputs (GPU enabled/disabled, allocation).
    *   **Mocks:** `tf.ConfigProto`, `tf.Session`, `set_session`, `os.environ`.
*   **`deactivate_gpu`**:
    *   **Test:** Verify `os.environ["CUDA_VISIBLE_DEVICES"]` is set to "-1".
    *   **Mocks:** `os.environ`.
*   **`can_use_gpu`**:
    *   **Test:** Mock `pip.get_installed_distributions` to return packages with and without `tensorflow-gpu`.

### 4. `python-lib/dfgenerator.py`

*   **`DfGenerator`**:
    *   **Test:** Initialize with a sample DataFrame. Verify `_read_classes` and `_read_data` correctly parse standard and JSON-formatted label data.
    *   **Mocks:** `keras_retinanet.preprocessing.csv_generator.Generator.__init__` (to avoid superclass complex init if needed).

### 5. `python-lib/retinanet_model.py`

*   **`get_model`**:
    *   **Test:** Verify correct model loading logic (CPU vs Multi-GPU).
    *   **Mocks:** `resnet50_retinanet`, `multi_gpu_model`, `tf.device`.
*   **`get_test_model`**:
    *   **Test:** Verify it calls `get_model` with `freeze=True` and wraps result in `retinanet_bbox`.
    *   **Mocks:** `get_model`, `retinanet_bbox`.
*   **`compile_model`**:
    *   **Test:** Verify `model.compile` is called with correct optimizer and loss functions.
    *   **Mocks:** `optimizers`, `keras_retinanet.losses`.
*   **`find_objects`**:
    *   **Test:** Verify image preprocessing, resizing, and batch prediction logic. Ensure bounding boxes are rescaled.
    *   **Mocks:** `read_image_bgr`, `preprocess_image`, `resize_image`, `model.predict_on_batch`.
*   **`detect_in_video_file`**:
    *   **Test:** Verify video processing loop (reading frames, detecting, writing).
    *   **Mocks:** `cv2.VideoCapture`, `cv2.VideoWriter`, `find_objects_single`, `misc_utils.mkv_to_mp4`.
*   **`get_random_augmentator`**:
    *   **Test:** Verify `random_transform_generator` is called with correct config values.
    *   **Mocks:** `random_transform_generator`.
*   **`MultiGPUModelCheckpoint`**:
    *   **Test:** Verify `on_epoch_end` swaps the model to `base_model` before saving.

### 6. `python-runnables/create-api-service/runnable.py`

*   **`get_params`**:
    *   **Test:** Verify validation logic for various config inputs (valid/invalid folder IDs, boolean flags).
    *   **Mocks:** `dataiku.ManagedFolder` (via `project.list_managed_folders`), `project.list_api_services`.
*   **`get_model_endpoint_settings`**:
    *   **Test:** Verify the returned dictionary structure matches the expected API service endpoint configuration.

## Prerequisites

*   Install `pytest` and `pytest-mock`.
*   Ideally, run tests in an environment where `dataiku` and `keras_retinanet` packages are mocked or available (if integration testing, but here we focus on unit testing with mocks).
