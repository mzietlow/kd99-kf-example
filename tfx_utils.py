from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs

NUMERICAL_KEYS = [f"num_{i}" for i in range(38)]
CATEGORICAL_KEYS = ["transport_protocol", "application_protocol", "cat_0", ]
LABEL_KEYS = ["label_0"]
VOCAB_SIZE = 100  # a lousy guess
OOV_SIZE = 10  # a lousy guess

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        dense_tensor = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value)
    else:
        dense_tensor = x

    return tf.squeeze(dense_tensor, axis=1)


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    print("Scale numerical keys to z-score. Add to output.")
    for key in NUMERICAL_KEYS:
        cache = tft.scale_to_z_score(_fill_in_missing(inputs[key]))
        outputs[key] = tft.scale_to_0_1(cache)
    print("Add categorical keys to output.")
    for key in CATEGORICAL_KEYS:
        outputs[key] = _fill_in_missing(inputs[key])
    print("Add label keys to output")
    for key in LABEL_KEYS:
        outputs[key] = inputs[key]
    return outputs


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key="tips")

    return dataset


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)
