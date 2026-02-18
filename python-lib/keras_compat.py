import importlib.metadata as importlib_metadata
import os
import sys


def _safe_version(pkg_name):
    try:
        return importlib_metadata.version(pkg_name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _ensure_initializer_alias(initializers, alias, target):
    if not hasattr(initializers, alias):
        setattr(initializers, alias, target)


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


def bootstrap_keras_retinanet_compat():
    """Install runtime shims required by legacy keras-retinanet builds."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

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
