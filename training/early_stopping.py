"""
training/early_stopping.py
--------------------------
Early stopping callback.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.debug("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
