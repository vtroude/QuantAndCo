import pandas   as pd

from abc    import ABC, abstractmethod


class StrategyAbstract(ABC):
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        pass