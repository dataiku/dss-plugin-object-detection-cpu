import logging
import os
os.environ["OPENCV_LOG_LEVEL"] = "DEBUG"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"   # important
import cv2
import math

import numpy as np
from keras_compat import bootstrap_keras_retinanet_compat

# Must run before importing tensorflow/keras/keras-retinanet symbols.
bootstrap_keras_retinanet_compat()

import tensorflow as tf
import keras
from keras import optimizers
from keras import callbacks
import inspect

try:
    from keras.utils import multi_gpu_model
except Exception:
    multi_gpu_model = None
import keras_retinanet
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.transform import random_transform_generator

import misc_utils


logging.basicConfig(level=logging.INFO, format='[Object Detection] %(levelname)s - %(message)s')


def _ensure_legacy_keras_initializers():
    """Backwards compatibility for older keras-retinanet code paths."""
    if hasattr(keras.initializers, 'normal'):
        return

    def _normal(mean=0.0, stddev=0.05, seed=None):
        return keras.initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

    keras.initializers.normal = _normal


def _to_numpy_dtype(dtype):
    """Convert tf/keras dtypes to numpy dtypes for old keras-retinanet code."""
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype
    try:
        return tf.as_dtype(dtype).as_numpy_dtype
    except Exception:
        pass
    try:
        return np.dtype(dtype)
    except Exception:
        return dtype


def _ensure_legacy_retinanet_prior_probability():
    """Patch keras-retinanet PriorProbability for keras/tf dtype compatibility."""
    prior_cls = getattr(keras_retinanet.initializers, 'PriorProbability', None)
    if prior_cls is None:
        return

    def _patched_call(self, shape, dtype=None):
        np_dtype = _to_numpy_dtype(dtype) or np.float32
        return np.ones(shape, dtype=np_dtype) * -math.log((1 - self.probability) / self.probability)

    prior_cls.__call__ = _patched_call


def _ensure_legacy_tf_resize_images():
    """Restore removed tf.image.resize_images symbol for legacy keras-retinanet."""
    if hasattr(tf.image, 'resize_images'):
        return

    legacy_resize = getattr(tf.compat.v1.image, 'resize_images', None)
    if legacy_resize is not None:
        tf.image.resize_images = legacy_resize
        return

    def _resize_images(images, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False,
                       preserve_aspect_ratio=False, name=None):
        del align_corners  # not supported by tf.image.resize in newer TF
        return tf.image.resize(
            images=images,
            size=size,
            method=method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            name=name
        )

    tf.image.resize_images = _resize_images


_ensure_legacy_keras_initializers()
_ensure_legacy_retinanet_prior_probability()
_ensure_legacy_tf_resize_images()


def get_model(weights, num_classes, freeze=False, n_gpu=None):
    """Return a RetinaNet model.

    Args:
        weights:     Initial weights.
        num_classes: Number of classes to detect.
        freeze:      Freeze the ResNet backbone.
        n_gpu:       Number of gpu, if above 1, will set up a multi gpu model.

    Returns:
        The model to save.
        The model to train.
    """
    multi_gpu = n_gpu is not None and n_gpu > 1

    modifier = freeze_model if freeze else None

    if multi_gpu:
        if multi_gpu_model is None:
            logging.warning('`keras.utils.multi_gpu_model` is unavailable. Falling back to single model.')
            model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
            model.load_weights(weights, by_name=True, skip_mismatch=True)
            return model, model
        logging.info('Loading model in multi gpu mode.')
        with tf.device('/cpu:0'):
            model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
            model.load_weights(weights, by_name=True, skip_mismatch=True)

        multi_model = multi_gpu_model(model, gpus=n_gpu)
        return model, multi_model
    elif n_gpu == 1:
        logging.info('Loading model in single gpu mode.')
    else:
        logging.info('Loading model in cpu mode. It will be slow, use gpu if possible!')

    model = resnet50_retinanet(num_classes=num_classes, modifier=modifier)
    model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model, model


def get_test_model(weights, num_classes):
    """Returns an inference retinanet model.

    Args:
        weights:     Initial weights.
        num_classes: Number of classes to detect.
        n_gpu:       Number of gpu, if above 1, will set up a multi gpu model.

    Returns:
        The inference model.
    """
    model = get_model(weights, num_classes, freeze=True, n_gpu=1)[0]
    test_model = retinanet_bbox(model=model)
    return test_model


def compile_model(model, configs):
    """Compile retinanet."""
    lr = float(configs['lr'])
    if configs['optimizer'].lower() == 'adam':
        try:
            opt = optimizers.Adam(learning_rate=lr, clipnorm=0.001)
        except Exception:
            opt = optimizers.adam(lr=lr, clipnorm=0.001)
    else:
        try:
            opt = optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True, clipnorm=0.001)
        except Exception:
            opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True, clipnorm=0.001)

    model .compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=opt
    )


def find_objects(model, paths):
    """Find objects with bach size >= 1.

    To support batch size > 1, this method implements a naive ratio grouping
    where batch of images will be processed together solely if they have a
    similar shape.

    Args:
        model: A retinanet model in inference mode.
        paths: Paths to all images to process. The *maximum* batch size is the
               number of paths.

    Returns:
        Boxes, scores, and labels.
        Their shapes: (b, 300, 4), (b, 300), (b, 300)
        With b the batch size.
    """
    if isinstance(paths, str):
        paths = [paths]

    path_i = 0
    nb_paths = len(paths)
    b_boxes, b_scores, b_labels = [], [], []

    while nb_paths != path_i:
        images = []
        scales = []
        previous_shape = None

        for path in paths[path_i:]:
            image = read_image_bgr(path)
            if previous_shape is not None and image.shape != previous_shape:
                break # Cannot make the batch bigger due to ratio difference

            previous_shape = image.shape
            path_i += 1

            image = preprocess_image(image)
            image, scale = resize_image(image)

            images.append(image)
            scales.append(scale)

        images = np.stack(images)
        boxes, scores, labels = model.predict_on_batch(images)

        for i, scale in enumerate(scales):
            boxes[i, :, :] /= scale # Taking in account the resizing factor

        b_boxes.append(boxes)
        b_scores.append(scores)
        b_labels.append(labels)



    b_boxes = np.concatenate(b_boxes, axis=0)
    b_scores = np.concatenate(b_scores, axis=0)
    b_labels = np.concatenate(b_labels, axis=0)

    return b_boxes, b_scores, b_labels


def find_objects_single(model, image, min_side=800, max_side=1333):
    """Short method to detect objects. Only supports batch size = 1."""
    if isinstance(image, str):
        image = read_image_bgr(image)
    else:
        image = image.copy()
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=min_side, max_side=max_side)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes[0, :, :] /= scale # Taking in account the resizing factor

    return boxes, scores, labels


def detect_in_video_file(model, in_vid_path, out_dir, detection_rate=None):
    """Detect objects in a video and write to the video frames.

    Args:
        model:       Trained model, must be in inference mode.
        in_vid_path: Video path.
        out_dir:     Folder where produced video will be stored.
        fps:         FPS of the produced video, it unspecified it will be the default video FPS.

    Returns:
        None
    """

    def probe_writer(out_dir, vid_name, w, h, fps):
        logging.info("probe_writer: dir=%s exists=%s writable=%s", out_dir, os.path.isdir(out_dir), os.access(out_dir, os.W_OK))
        logging.info("probe_writer: size=(%s,%s) fps=%s", w, h, fps)

        tests = [
            (cv2.CAP_FFMPEG, "XVID", os.path.join(out_dir, f"{vid_name}-detected.avi")),
            (cv2.CAP_FFMPEG, "MJPG", os.path.join(out_dir, f"{vid_name}-detected.avi")),
            (cv2.CAP_GSTREAMER, "XVID", os.path.join(out_dir, f"{vid_name}-detected.avi")),
        ]

        for api, codec, path in tests:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vw = cv2.VideoWriter(path, api, fourcc, float(fps), (int(w), int(h)))
            ok = vw.isOpened()
            logging.info("probe_writer: api=%s codec=%s path=%s opened=%s", api, codec, path, ok)
            if ok:
                try:
                    logging.info("probe_writer: backend=%s", vw.getBackendName())
                except Exception:
                    pass
                return vw, path, codec, api

        return None, None, None, None
    vid_name = os.path.splitext(os.path.basename(in_vid_path))[0]
    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened(), f"Cannot open input video: {in_vid_path}"

    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    vw, out_path, codec, api = probe_writer(out_dir, vid_name, vid_width, vid_height, fps)
    if vw is None:
        cap.release()
        raise RuntimeError("VideoWriter could not be opened with tested APIs/codecs. Check OpenCV build/backends in this env.")

    logging.info("Writing output: %s (codec=%s api=%s)", out_path, codec, api)

    idx = 0
    boxes = scores = labels = None
    detection_rate = max(1, int(detection_rate or 1))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        if idx % detection_rate == 0:
            boxes, scores, labels = find_objects_single(model, img)

        if boxes is not None:
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < 0.5:
                    break
                misc_utils.draw_box(img, box, color=(0, 0, 255))

        vw.write(img)
        idx += 1

    cap.release()
    vw.release()

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Output file missing/empty after write: {out_path}")

    logging.info("Done. Frames written: %d, output: %s, size=%d bytes", idx, out_path, os.path.getsize(out_path))

    misc_utils.source_to_mp4(out_path, remove_mkv=True, has_audio=False, quiet=True)


def get_random_augmentator(configs):
    """Return a retinanet data augmentator. @config comes the recipe params."""
    return random_transform_generator(
        min_rotation    = float(configs['min_rotation']),
        max_rotation    = float(configs['max_rotation']),
        min_translation = (float(configs['min_trans']), float(configs['min_trans'])),
        max_translation = (float(configs['max_trans']), float(configs['max_trans'])),
        min_shear       = float(configs['min_shear']),
        max_shear       = float(configs['max_shear']),
        min_scaling     = (float(configs['min_scaling']), float(configs['min_scaling'])),
        max_scaling     = (float(configs['max_scaling']), float(configs['max_scaling'])),
        flip_x_chance   = float(configs['flip_x']),
        flip_y_chance   = float(configs['flip_y'])
    )


def get_model_checkpoint(path, base_model, n_gpu):
    if n_gpu <= 1:
        return callbacks.ModelCheckpoint(path, verbose=0, save_best_only=True, save_weights_only=True)

    return MultiGPUModelCheckpoint(path, base_model, verbose=0, save_best_only=True, save_weights_only=True)


class MultiGPUModelCheckpoint(callbacks.ModelCheckpoint):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        kwargs = dict(
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
        )
        if 'period' in inspect.signature(callbacks.ModelCheckpoint.__init__).parameters:
            kwargs['period'] = period

        super(MultiGPUModelCheckpoint, self).__init__(filepath, **kwargs)
        self.base_model = base_model

    def on_epoch_end(self, epoch, logs=None):
        # Must behave like ModelCheckpoint on_epoch_end but save base_model instead

        # First retrieve model
        model = self.model

        # Then switching model to base model
        self.model = self.base_model

        # Calling super on_epoch_end
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # Resetting model afterwards
        self.model = model
