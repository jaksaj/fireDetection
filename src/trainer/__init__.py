"""Training orchestrators for all project iterations."""

from src.trainer.base import BaseTrainer, EpochMetrics
from src.trainer.binary import BinaryTrainer, Trainer
from src.trainer.multiclass import MulticlassTrainer
from src.trainer.robust import RobustMulticlassTrainer
from src.trainer.segmentation import SegmentationTrainer

__all__ = [
    "BaseTrainer",
    "BinaryTrainer",
    "EpochMetrics",
    "MulticlassTrainer",
    "RobustMulticlassTrainer",
    "SegmentationTrainer",
    "Trainer",
]
