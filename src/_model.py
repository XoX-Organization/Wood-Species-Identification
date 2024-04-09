
from loguru import logger
from pathlib import Path
from tensorflow.keras.models import (
    Model,
    load_model,
)
from typing import List, Tuple
from uuid import uuid4


class ModelContext:
    def __init__(self, model: Model, path: Path):
        self._model: Model = model
        self._path: Path = path

    @property
    def model(self) -> Model:
        return self._model

    @property
    def filename(self) -> str:
        return self._path.name

    def delete(self) -> None:
        self._path.unlink(missing_ok=True)
        logger.info(f"Model deleted: {self._path}")

    def save(self) -> None:
        self._model.save(self._path)
        logger.debug(f"Model saved to {self._path}")


class ModelFactory:
    MODEL_DIRNAME = "models"
    MODEL_DIRPATH = Path(MODEL_DIRNAME).resolve()

    @staticmethod
    def _mkdir() -> None:
        ModelFactory.MODEL_DIRPATH.mkdir(exist_ok=True)

    @staticmethod
    def _create_model_filename(model: Model) -> str:
        return ModelFactory.MODEL_DIRPATH / f"{model.name}-{str(uuid4()).lower()[:6]}.keras"

    @staticmethod
    def create(model: Model=None) -> ModelContext:
        if model is None:
            model = Model()

        ModelFactory._mkdir()
        return ModelContext(model, Path(ModelFactory._create_model_filename(model)))

    @staticmethod
    def models() -> List[ModelContext]:
        ModelFactory._mkdir()
        model_list: List[Tuple[Model, Path]] = []

        for f in ModelFactory.MODEL_DIRPATH.iterdir():
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
