"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .utils import _reduce

@dataclass
class ClassificationMetrics:
    Acc: Union[float, np.ndarray] = None
    F1: Union[float, np.ndarray] = None


def ACC(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    # Ensure y and y_hat have the same shape
    assert y.shape == y_hat.shape, "Shape of y and y_hat must be the same"

    # Calculate accuracy
    correct = (y == y_hat)
    if axis is not None:
        accuracy = np.mean(correct, axis=axis)
    else:
        accuracy = np.mean(correct)

    # Calculate F1 score
    if axis is not None:
        f1 = np.apply_along_axis(lambda x: f1_score(y[:, x], y_hat[:, x], average='weighted'), axis,
                                 np.arange(y.shape[axis]))
    else:
        f1 = f1_score(y, y_hat, average='weighted')

    # Apply reduction
    if reduction == "mean":
        return np.mean(accuracy), np.mean(f1)
    elif reduction == "sum":
        return np.sum(accuracy), np.sum(f1)
    elif reduction is None:
        return accuracy, f1
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Expected 'mean', 'sum', or None.")


def get_classification_metrics(
    y: npt.NDArray,
    y_hat: npt.NDArray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    accuracy, f1 = ACC(y=y, y_hat=y_hat, axis=axis, reduction=reduction)
    return ClassificationMetrics(
        Acc=accuracy,
        F1=f1,
    )
