try:
    import importlib.metadata as importlib_metadata
except Exception:
    try:
        import importlib_metadata  # type: ignore
    except Exception:
        importlib_metadata = None
import os
import sys
import types


def _safe_version(pkg_name):
    if importlib_metadata is not None:
        try:
            return importlib_metadata.version(pkg_name)
        except Exception:
            pass

    try:
        import pkg_resources
        return pkg_resources.get_distribution(pkg_name).version
    except Exception:
        return "not-installed"


def _ensure_initializer_alias(initializers, alias, target):
    if not hasattr(initializers, alias):
        setattr(initializers, alias, target)


def _enable_legacy_tf_execution_mode():
    """Force TF2 runtime into TF1-style execution for legacy keras-retinanet."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "Could not import tensorflow while preparing keras-retinanet compatibility."
        ) from exc

    # keras-retinanet 0.5.x uses graph-oriented control-flow patterns.
    if tf.executing_eagerly():
        try:
            tf.compat.v1.disable_eager_execution()
        except Exception:
            # If eager is already initialized, continue and rely on remaining shims.
            pass

    disable_cf_v2 = getattr(tf.compat.v1, "disable_control_flow_v2", None)
    if callable(disable_cf_v2):
        try:
            disable_cf_v2()
        except Exception:
            pass


def _patch_keras_retinanet_initializers():
    try:
        import keras_retinanet.initializers as kr_initializers
    except Exception as exc:
        raise RuntimeError(
            "Could not import keras_retinanet.initializers to apply dtype compatibility patch."
        ) from exc

    prior_probability = getattr(kr_initializers, "PriorProbability", None)
    if prior_probability is None:
        raise RuntimeError(
            "keras_retinanet.initializers.PriorProbability was not found; "
            "cannot apply TF dtype compatibility patch."
        )

    original_call = getattr(prior_probability, "__call__", None)
    if original_call is None:
        raise RuntimeError(
            "PriorProbability.__call__ was not found; cannot apply dtype compatibility patch."
        )

    if getattr(original_call, "__dss_dtype_patch__", False):
        return

    def _patched_call(self, shape, dtype=None):
        if dtype is not None and hasattr(dtype, "as_numpy_dtype"):
            dtype = dtype.as_numpy_dtype
        return original_call(self, shape, dtype=dtype)

    _patched_call.__dss_dtype_patch__ = True
    prior_probability.__call__ = _patched_call


def _has_real_keras_retinanet_package():
    """Return True when keras_retinanet is loaded as an actual package."""
    kr_module = sys.modules.get("keras_retinanet")
    if kr_module is None:
        return True
    if not isinstance(kr_module, types.ModuleType):
        return False
    return hasattr(kr_module, "__path__")


def _patch_keras_retinanet_map_fn():
    """Patch keras-retinanet TF backend map_fn for TF2 signature compatibility."""
    try:
        import tensorflow as tf
        import keras_retinanet.backend.tensorflow_backend as tf_backend
    except Exception as exc:
        raise RuntimeError(
            "Could not import keras_retinanet.backend.tensorflow_backend for map_fn patch."
        ) from exc

    original_map_fn = getattr(tf_backend, "map_fn", None)
    if original_map_fn is None:
        raise RuntimeError("keras_retinanet.backend.tensorflow_backend.map_fn was not found.")

    if getattr(original_map_fn, "__dss_map_fn_patch__", False):
        return

    def _patched_map_fn(*args, **kwargs):
        if "dtype" in kwargs and "fn_output_signature" not in kwargs:
            kwargs["fn_output_signature"] = kwargs.pop("dtype")
        return tf.map_fn(*args, **kwargs)

    _patched_map_fn.__dss_map_fn_patch__ = True
    tf_backend.map_fn = _patched_map_fn


def _patch_keras_retinanet_filter_detections():
    """Patch legacy filter_detections for TF2 while/map dynamic-shape compatibility."""
    try:
        import tensorflow as tf
        import keras
        import keras_retinanet.layers.filter_detections as fd
    except Exception as exc:
        raise RuntimeError(
            "Could not import keras_retinanet.layers.filter_detections for compatibility patch."
        ) from exc

    original_fn = getattr(fd, "filter_detections", None)
    if original_fn is None:
        raise RuntimeError("keras_retinanet.layers.filter_detections.filter_detections was not found.")

    if getattr(original_fn, "__dss_filter_detections_patch__", False):
        return

    backend_ones = keras.backend.ones

    def _patched_filter_detections(*args, **kwargs):
        def _compat_ones(shape, dtype=None, name=None):
            # Legacy code path uses keras.backend.ones((dynamic_tensor,), dtype=...)
            # inside map_fn/while loops, which fails in modern TF graph control flow.
            if isinstance(shape, (tuple, list)) and len(shape) == 1 and tf.is_tensor(shape[0]):
                dim = tf.cast(shape[0], tf.int32)
                out_shape = tf.reshape(dim, [1])
                target_dtype = tf.as_dtype(dtype or keras.backend.floatx())
                return tf.fill(out_shape, tf.cast(1, target_dtype), name=name)
            return backend_ones(shape=shape, dtype=dtype, name=name)

        previous_ones = keras.backend.ones
        keras.backend.ones = _compat_ones
        try:
            return original_fn(*args, **kwargs)
        finally:
            keras.backend.ones = previous_ones

    patched_fn = tf.autograph.experimental.do_not_convert(_patched_filter_detections)
    patched_fn.__dss_filter_detections_patch__ = True
    fd.filter_detections = patched_fn


def _patch_keras_backend_ones():
    """Allow keras.backend.ones((tensor,), ...) in TF while/map control-flow."""
    try:
        import tensorflow as tf
        import keras
    except Exception as exc:
        raise RuntimeError("Could not import tensorflow/keras for backend.ones patch.") from exc

    backend_ones = getattr(keras.backend, "ones", None)
    if backend_ones is None:
        raise RuntimeError("keras.backend.ones was not found.")

    if getattr(backend_ones, "__dss_ones_shape_patch__", False):
        return

    def _patched_ones(shape, dtype=None, name=None):
        if isinstance(shape, (tuple, list)):
            dynamic_dims = [dim for dim in shape if tf.is_tensor(dim)]
            if dynamic_dims:
                shape = tf.stack([tf.cast(dim, tf.int32) for dim in shape], axis=0)
        return backend_ones(shape=shape, dtype=dtype, name=name)

    _patched_ones.__dss_ones_shape_patch__ = True
    keras.backend.ones = _patched_ones


def bootstrap_keras_retinanet_compat():
    """Install runtime shims required by legacy keras-retinanet builds."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    _enable_legacy_tf_execution_mode()

    try:
        import tf_keras as legacy_keras
        # Force all downstream "import keras" calls to use the legacy Keras 2 API.
        sys.modules["keras"] = legacy_keras
    except Exception:
        pass

    try:
        import keras
    except Exception as exc:
        raise RuntimeError(
            "Could not import keras while preparing keras-retinanet compatibility. "
            "Ensure a TensorFlow 2.x + tf-keras compatible stack is installed."
        ) from exc

    try:
        _ensure_initializer_alias(keras.initializers, "normal", keras.initializers.RandomNormal)
        _ensure_initializer_alias(keras.initializers, "uniform", keras.initializers.RandomUniform)
    except Exception as exc:
        raise RuntimeError(
            "Failed to install keras initializer compatibility aliases for keras-retinanet."
        ) from exc

    missing_symbols = [
        symbol for symbol in ("normal", "uniform") if not hasattr(keras.initializers, symbol)
    ]
    if missing_symbols:
        tf_version = _safe_version("tensorflow")
        keras_version = _safe_version("keras")
        tf_keras_version = _safe_version("tf-keras")
        raise RuntimeError(
            "Unsupported Keras compatibility state for keras-retinanet. "
            "Missing keras.initializers symbols: {}. "
            "Installed versions -> tensorflow={}, keras={}, tf-keras={}. "
            "Use TensorFlow 2.x with TF_USE_LEGACY_KERAS=1 and tf-keras installed."
            .format(", ".join(missing_symbols), tf_version, keras_version, tf_keras_version)
        )

    # Unit tests can inject lightweight stubs into sys.modules for keras-retinanet.
    # In that case, skip runtime patching that requires real package internals.
    if not _has_real_keras_retinanet_package():
        return

    try:
        _patch_keras_retinanet_initializers()
    except Exception as exc:
        tf_version = _safe_version("tensorflow")
        keras_version = _safe_version("keras")
        tf_keras_version = _safe_version("tf-keras")
        numpy_version = _safe_version("numpy")
        raise RuntimeError(
            "Failed to patch keras_retinanet.initializers.PriorProbability for TF dtype compatibility. "
            "Installed versions -> tensorflow={}, keras={}, tf-keras={}, numpy={}. "
            "Ensure keras-retinanet is installed and compatible with this runtime."
            .format(tf_version, keras_version, tf_keras_version, numpy_version)
        ) from exc

    try:
        _patch_keras_retinanet_map_fn()
        _patch_keras_retinanet_filter_detections()
        _patch_keras_backend_ones()
    except Exception as exc:
        tf_version = _safe_version("tensorflow")
        keras_version = _safe_version("keras")
        tf_keras_version = _safe_version("tf-keras")
        raise RuntimeError(
            "Failed to patch keras-retinanet filter_detections/map_fn compatibility. "
            "Installed versions -> tensorflow={}, keras={}, tf-keras={}. "
            "Ensure legacy keras-retinanet code paths can run on this TensorFlow runtime."
            .format(tf_version, keras_version, tf_keras_version)
        ) from exc
