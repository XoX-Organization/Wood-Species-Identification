
from keras_tuner import (
    Hyperband,
    HyperParameters,
)
from loguru import logger
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.data import (
    AUTOTUNE,
    Dataset,
)
from tensorflow import keras
from tensorflow.keras import (
    Model,
    regularizers,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import (
    clone_model,
    Sequential,
)
from tensorflow.keras.optimizers import Adam
from typing import (
    Any,
    Callable,
    List,
    Optional,
)

from src import (
    Dataset as WSI_Dataset,
    ModelContext,
    ModelFactory,
)

import itertools
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


IMG_WIDTH = 200
IMG_HEIGHT = 200


# The name of the model to be trained, should not include the file extension
# None if a new model is to be trained
OVERWRITE_MODEL: Optional[str] = None


# Set to True if data augmentation is to be used
DATA_AUGMENTATION: bool = False
# The `DATA_AUGMENTATION_LAYERS` will be used only if `DATA_AUGMENTATION` is True
DATA_AUGMENTATION_LAYERS = [
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
]


# Variables for the model
BATCH_SIZE: int = 4
VALIDATION_SPLIT: float = 0.3


# The number of neurons in the final layer, should be 12 for this dataset
FINAL_LAYER_UNITS: int = 12


def HYPERMODEL_CREATION_CALLBACK(
    hp: HyperParameters,
    *,
    model: Optional[Model]=None,
) -> Model:
    """The function to create a hypermodel
    """
    if model is None:
        model = Sequential(
            [
                Rescaling(1. / 255),
                Conv2D(
                    filters=hp.Int("conv_1_filter", min_value=32, max_value=128, step=16),
                    kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
                    kernel_regularizer=regularizers.l2(hp.Choice("conv_1_l2", values=[1e-3, 1e-4, 1e-5, 1e-6])),
                    activation="relu",
                    input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3),
                ),
                MaxPooling2D(
                ),
                Dropout(
                    rate=hp.Float("dropout_1_rate", min_value=0.1, max_value=0.8, step=0.1),
                ),
                Conv2D(
                    filters=hp.Int("conv_2_filter", min_value=32, max_value=128, step=16),
                    kernel_regularizer=regularizers.l2(hp.Choice("conv_2_l2", values=[1e-3, 1e-4, 1e-5, 1e-6])),
                    kernel_size=hp.Choice("conv_2_kernel", values=[3, 5]),
                    activation="relu",
                ),
                MaxPooling2D(
                ),
                Dropout(
                    rate=hp.Float("dropout_2_rate", min_value=0.1, max_value=0.8, step=0.1),
                ),
                Flatten(
                    input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3)
                ),
                Dense(
                    units=hp.Int("dense_1_units", min_value=32, max_value=512, step=32),
                    kernel_regularizer=regularizers.l2(hp.Choice("dense_1_l2", values=[1e-3, 1e-4, 1e-5, 1e-6])),
                    activation="relu",
                ),
                Dropout(
                    rate=hp.Float("dropout_3_rate", min_value=0.1, max_value=0.5, step=0.1),
                ),
                Dense(
                    units=FINAL_LAYER_UNITS,
                ),
            ]
        )

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss=SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            "accuracy",
        ]
    )

    return model


def FIT_CALLBACKS(model_name: str) -> List[Callable]:
    """The callbacks to be called after done of each epoch
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
        ),
        ModelCheckpoint(
            filepath=f"caches/checkpoints/{model_name}.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        TensorBoard(
            log_dir=f"logs/fit/{model_name}",
            histogram_freq=1,
            profile_batch=0,
        ),
    ]


_raw_train_ds, _raw_val_ds, _raw_test_ds = WSI_Dataset.get(validation_split=VALIDATION_SPLIT)

class_names = _raw_train_ds.class_names

logger.info(f"Raw train set with {len(_raw_train_ds)} samples and {len(_raw_train_ds.class_names)} of classes, which are {', '.join(_raw_train_ds.class_names)}")
logger.info(f"Raw validation set with {len(_raw_val_ds)} samples and {len(_raw_val_ds.class_names)} of classes, which are {', '.join(_raw_val_ds.class_names)}")
logger.info(f"Raw test set with {len(_raw_test_ds)} samples and {len(_raw_test_ds.class_names)} of classes, which are {', '.join(_raw_test_ds.class_names)}")


def _process_ds(ds: Dataset, batch: int, shuffle: bool) -> Dataset:
    ds = ds.batch(batch)

    if shuffle:
        ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)

    ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def _augment(ds: Dataset) -> Dataset:
    data_augmentation = Sequential(DATA_AUGMENTATION_LAYERS)

    result = ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    return result

if DATA_AUGMENTATION:
    _raw_train_ds = _augment(_raw_train_ds)
    _raw_val_ds = _augment(_raw_val_ds)
    logger.info("Dataset has been augmented")


train_ds = _process_ds(_raw_train_ds, batch=BATCH_SIZE, shuffle=True)
val_ds = _process_ds(_raw_val_ds, batch=BATCH_SIZE, shuffle=False)
test_ds = _process_ds(_raw_test_ds, batch=1, shuffle=False)

logger.info(f"Batched train set with {len(train_ds)} samples")
logger.info(f"Batched validation set with {len(val_ds)} samples")
logger.info(f"Test set with {len(test_ds)} samples")


if OVERWRITE_MODEL is not None:
    _contexts: List[ModelContext] = ModelContext.models()
    context: Optional[ModelContext] = next(
        filter(lambda x: x.name == OVERWRITE_MODEL, _contexts),
        None
    )

    if context is None:
        raise ValueError(f"Model {OVERWRITE_MODEL} not found")

    logger.info(f"Model {context.name} will be used for this training")


else:
    context = None


initial_model: Optional[Model] = clone_model(context.model) if context is not None else None


tuner = Hyperband(
    lambda hp: HYPERMODEL_CREATION_CALLBACK(
        hp,
        model=initial_model,
    ),
    objective="val_accuracy",
    max_epochs=200,
    factor=3,
    directory="caches",
    project_name="hyperband",
)


tuner.search(
    train_ds,
    epochs=200,
    validation_data=val_ds,
    callbacks=FIT_CALLBACKS("hyperband"),
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

logger.debug(f"Best HPs: {best_hps}")
logger.success(
    f"The hyperparameter search is complete. The optimal values are\n" + \
    "\n".join([f"{k.capitalize():25s}: {v}" for k, v in best_hps.values.items()])
)


__model: Model = tuner.hypermodel.build(best_hps)


if context is not None:
    context.model = __model
    logger.info(f"Re-using the model {context.name}")

else:
    context = ModelFactory.create(__model)
    logger.info(f"Created a new model {context.name}")


context.model.summary(
    expand_nested=True,
)

history = context.model.fit(
    train_ds,
    callbacks=FIT_CALLBACKS(context.name),
    validation_data=val_ds,
    epochs=10000,
    verbose=1,
)


eval_result = context.model.evaluate(test_ds)
logger.info(f"Test loss: {eval_result[0]}")
logger.info(f"Test accuracy: {eval_result[1]}")

best_epoch = history.history["val_accuracy"].index(
    max(history.history["val_accuracy"])
) + 1


logger.debug(f"Best epoch: {best_epoch}")
logger.info(f"Re-instantiate the hypermodel and train it with the optimal number of epochs {best_epoch}.")


context.model = tuner.hypermodel.build(best_hps)

context.model.fit(
    train_ds,
    callbacks=FIT_CALLBACKS(context.name),
    validation_data=val_ds,
    epochs=best_epoch,
    verbose=1,
)

eval_result = context.model.evaluate(test_ds)
logger.info(f"Test loss: {eval_result[0]}")
logger.info(f"Test accuracy: {eval_result[1]}")


predictions = context.model.predict(
    test_ds,
    verbose=1,
)

logger.debug(f"Predictions shape: {predictions.shape}")
logger.debug(f"Predictions\n{predictions}")


actual = np.array([l.numpy() for _, l in test_ds])
predicted = np.argmax(predictions, axis=-1)

logger.debug(f"Actual shape: {actual.shape}")
logger.debug(f"Actual values\n{actual}")

logger.debug(f"Predicted shape: {predicted.shape}")
logger.debug(f"Predicted values\n{predicted}")


logger.info(f"Accuracy: {accuracy_score(actual, predicted)}")
logger.info(f"Precision: {precision_score(actual, predicted, average='micro')}")
logger.info(f"Sensitivity recall: {recall_score(actual, predicted, average='micro')}")
logger.info(f"Specificity: {recall_score(actual, predicted, pos_label=0, average='micro')}")
logger.info(f"F1 score: {f1_score(actual, predicted, average='micro')}")


cm = confusion_matrix(actual, predicted)

logger.debug(f"Confusion Matrix\n{cm}")


context.model.summary()
context.save()
