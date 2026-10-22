from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, sample):
        pass

    @abstractmethod
    def train_only(self) -> bool:
        pass
