
from loguru import logger
from pathlib import Path
from tensorflow.keras.models import (
    Model,
    load_model,
)
from time import time
from typing import List, Tuple
from uuid import uuid4


class ModelContext:
    """A context manager for TensorFlow Keras models, handling model
    operations such as saving, deleting, and property management.
    """
    def __init__(self, model: Model, path: Path):
        self._model: Model = model
        self._path: Path = path

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, other: Model) -> None:
        """Sets a new model, backing up the current one before replacement.
        """
        ModelFactory.make_backup(self)
        self._model = other

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def filename(self) -> str:
        return self._path.name

    def delete(self) -> None:
        """Deletes the model file from the filesystem.
        """
        self._path.unlink(missing_ok=True)
        logger.info(f"Model deleted: {self._path}")

    def save(self) -> None:
        """Saves the model to its file path.
        """
        self._save(self._path)

    def _save(self, path: Path) -> None:
        """Helper method to save the model to a specified path.
        """
        self._model.save(path)
        logger.debug(f"Model saved to {path}")


class ModelFactory:
    """A factory class for creating and managing ModelContext objects, facilitating
    model file operations such as creation, backup, and retrieval.
    """
    MODEL_DIRNAME = "models"
    MODEL_DIRPATH = Path(MODEL_DIRNAME).resolve()
    MODEL_BACKUP_DIRNAME = "backups"
    MODEL_BACKUP_DIRPATH = MODEL_DIRPATH.joinpath(MODEL_BACKUP_DIRNAME)

    @classmethod
    def _mkdir(cls) -> None:
        cls.MODEL_DIRPATH.mkdir(exist_ok=True)
        cls.MODEL_BACKUP_DIRPATH.mkdir(exist_ok=True)

    @classmethod
    def _create_model_filename(cls, model: Model) -> str:
        return cls.MODEL_DIRPATH / f"{model.name}-{str(uuid4()).lower()[:6]}.keras"

    @classmethod
    def create(cls, model: Model=None) -> ModelContext:
        """Creates a new ModelContext with an optional initial model.
        """
        if model is None:
            model = Model()

        cls._mkdir()
        return ModelContext(model, Path(cls._create_model_filename(model)))

    @classmethod
    def models(cls) -> List[ModelContext]:
        """Retrieves all models from the storage directory as ModelContext objects.
        """
        cls._mkdir()
        model_list: List[Tuple[Model, Path]] = []

        for f in cls.MODEL_DIRPATH.iterdir():
            if not f.endswith(".keras"):
                continue

            try:
                model_list.append(
                    (load_model(f), f)
                )

            except Exception as e:
                logger.error(f"Error loading model {f}: {e}")

        return [
            ModelContext(model, path)
            for model, path in model_list
        ]

    @classmethod
    def make_backup(cls, context: ModelContext) -> None:
        """Creates a backup of the model associated with a given ModelContext.
        """
        bak_filename = f"{context.name}-{int(time())}.keras"
        context._save(MODEL_BACKUP_DIRPATH / bak_filename)
