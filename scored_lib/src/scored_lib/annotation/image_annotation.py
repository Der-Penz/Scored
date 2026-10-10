from dataclasses import dataclass, field
from time import time

from dataclasses_json import DataClassJsonMixin, dataclass_json

from scored_lib.annotation.keypoints import DartKeypoints
from scored_lib.annotation.throw_annotation import AnnotatedThrow
from scored_lib.dart.dart_throw import DartThrow


@dataclass_json
@dataclass(eq=True)
class ImageAnnotation(DataClassJsonMixin):
    """Annotation of a single training sample, which is a single image with its associated metadata.

    One annotation is stored per sample inside a leg directory, in an
    ``annotation.json`` file that sits next to the camera frame the annotation
    was made on.

    The metadata says which throw the sample was recorded for, ``throws`` holds
    every dart visible in the frame: the recorded throw first, followed by the
    ones that were copied or drawn on top of it.

    Parameters
    ----------
    round : int
        One-based index of the round (visit) within the leg.
    throw : int
        One-based index of the throw within the round (1, 2, or 3).
    timestamp : float
        POSIX timestamp recording when the throw occurred.
    is_bust : bool
        Flag indicating if this throw resulted in a bust, by default False.
    throws : list[AnnotatedThrow]
        The darts labeled in the linked camera frame, by default empty.
    """

    round: int
    throw: int
    timestamp: float = field(default_factory=lambda: time())
    is_bust: bool = False
    throws: list[AnnotatedThrow] = field(default_factory=list)

    @property
    def index(self) -> tuple[int, int]:
        """The one-based ``(round, throw)`` coordinates of this annotation.

        Returns
        -------
        tuple[int, int]
            The round index and the throw index within that round.
        """
        return (self.round, self.throw)
    
    def add_throw(self, dart_throw: DartThrow, keypoints: DartKeypoints | None) -> None:
        """Add a new throw to the annotation.

        Parameters
        ----------
        dart_throw : DartThrow
            The dart throw to add.
        keypoints : DartKeypoints | None, optional
            The keypoints of the dart, by default None.
        """
        annotated_throw = AnnotatedThrow(dart_throw, keypoints)
        self.throws.append(annotated_throw)
        
    def remove_throw(self, annotated_throw: AnnotatedThrow) -> None:
        """Remove a throw from the annotation.

        Parameters
        ----------
        annotated_throw : AnnotatedThrow
            The annotated throw to remove.
        """
        self.throws = [t for t in self.throws if t != annotated_throw]