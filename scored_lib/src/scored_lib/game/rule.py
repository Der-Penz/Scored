from enum import Enum


class FinishRule(Enum):
    """Rule that must be satisfied on the final scoring throw."""

    ANY = "any"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    DOUBLE_OR_BULL = "double_or_bull"


class StartRule(Enum):
    """Rule that must be satisfied before the leg starts scoring."""

    ANY = "any"
    DOUBLE = "double"
    TRIPLE = "triple"
