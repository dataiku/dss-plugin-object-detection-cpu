import os

import tensorflow as tf

try:
    # Keras 2.2.x
    from keras.backend.tensorflow_backend import set_session as keras_set_session
except Exception:
    try:
        # TF2 compatibility path
        from tensorflow.compat.v1.keras.backend import set_session as keras_set_session
    except Exception:
        keras_set_session = None


def _is_tf1():
    return tf.__version__.startswith('1.')


def _set_session(config):
    if keras_set_session is None:
        return
    keras_set_session(tf.compat.v1.Session(config=config))


def load_gpu_options(should_use_gpu, list_gpu, gpu_allocation):
    """Set up gpu configuration.
    
    Args:
        should_use_gpu: Boolean for gpu activation.
        list_gpu:       String of gpu uid separed by a comma.
        gpu_allocation: Gpu allocation percent.
        
    Returns:
        The gpu configurations.
    """
    gpu_options = {}
    if should_use_gpu:
        gpu_options['n_gpu'] = len(list_gpu.split(','))

        os.environ["CUDA_VISIBLE_DEVICES"] = list_gpu.strip()

        if _is_tf1():
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = list_gpu.strip()
            config.gpu_options.per_process_gpu_memory_fraction = gpu_allocation
            _set_session(config)
        else:
            # TF2/Keras3: keep visible devices via env var and allow dynamic growth.
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        deactivate_gpu()
        gpu_options['n_gpu'] = 0

    return gpu_options


def deactivate_gpu():
    """Disable gpu."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    
def can_use_gpu():
    """Check that system supports gpu."""
    # TF1 often used tensorflow-gpu, while TF2 usually ships GPU support in tensorflow.
    try:
        return len(tf.config.list_physical_devices('GPU')) > 0
    except Exception:
        return False


def set_gpus(gpus):
    """Short method to set gpu configuration."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus.strip()
    if _is_tf1():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = gpus.strip()
        _set_session(config)
