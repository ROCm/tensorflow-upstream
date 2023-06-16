# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU embeddings API."""

from typing import Any, Dict, Iterable, Optional, Text, Union

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v1
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("tpu.experimental.embedding.GPUEmbeddingV0")
class GPUEmbeddingV0(tpu_embedding_v1.TPUEmbeddingV0):
  """GPU embeddings API.

  NOTE: This API is not intended for large embedding table lookup.
  Embedding tables will be replicated across devices rather than sharding
  across them. To do large embedding table lookup, please use the
  `tpu.experimental.embedding.GPUEmbedding` class. This class is an alternative
  way to do embedding lookups on GPUs. See
  `tpu.experimental.tpu_hardware_feature.embedding_feature` for a detailed
  explanation.

  This class has to be created under the `MirroredStrategy`, Otherwise a RuntimeError
  will be raised.
  ```python
  strategy = tf.distribute.MirroredStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.GPUEmbeddingV0(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```
  When creating a distributed dataset that is to be passed to the lookup
  operation a special input option must be specified:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_fetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  Below is an example of a training and evaluation step:

  ```python
  optimizer = tf.keras.optimizers.SGD(0.1)

  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(embedding_features):
      with tf.GradientTape() as tape:
        tape.watch(embedding.embedding_table.values())
        activation = embedding(embedding_features)
        model_output = model(activations)
        loss = ...  # some function of labels and model_output

      embedding_gradients = tape.gradient(loss,
                                          embedding.embedding_table.values())
      optimizer.apply_gradients(list(zip(gradients,
                                mid_level_api.embedding_tables.values())))
      # Insert your model gradient and optimizer application here

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))

  @tf.function
  def evalution_step(dataset_iterator, num_steps):
    def tpu_step(embedding_features):
      activations = embedding(embedding_features)
      model_output = model(activations)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))
  ```
  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer]):  # pylint:disable=protected-access
    super(tpu_embedding_v1.TPUEmbeddingV0, self).__init__(feature_config, optimizer)
    self._strategy = distribute_lib.get_strategy()
    self._built = False

