import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def mount(self) -> None:
        pass

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        free_cols = [c for c in range(s.shape[1]) if s[0, c] == 0]
        return np.random.choice(free_cols)
