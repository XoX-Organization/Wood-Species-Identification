
from ._dataset import Dataset
from ._model import ModelContext, ModelFactory

from loguru import logger as _logger

import tensorflow as _tf


_gpus = _tf.config.experimental.list_physical_devices("GPU")

if len(_gpus) > 0:
    _tf.config.experimental.set_memory_growth(_gpus[0], True)
    _logger.info(f"GPU memory growth enabled for {_gpus[0]}")

else:
    _logger.warning("No GPU found, using CPU instead, GPU is highly recommended")
