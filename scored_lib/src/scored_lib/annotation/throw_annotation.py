from __future__ import annotations

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin, dataclass_json

from scored_lib.annotation.keypoints import DartKeypoints
from scored_lib.dart.dart_throw import DartThrow


@dataclass_json
@dataclass(frozen=True, eq=True)
class AnnotatedThrow(DataClassJsonMixin):
    """Hold throw data and keypoints for a single dart.

    Parameters
    ----------
    throw_data : DartThrow
        The score the dart scored, None for darts that carry no known score,
        e.g. a leftover dart of an earlier visit.
    keypoints : DartKeypoints | None
        The labeled tip and flight of this dart, by default unlabeled.
    """

    throw_data: DartThrow
    keypoints: DartKeypoints | None = None

    def make_copy(self) -> AnnotatedThrow:
        """Make a copy of this AnnotatedThrow."""

        if self.keypoints is not None:
            keypoints = DartKeypoints(
                tip=self.keypoints.tip,
                flight=self.keypoints.flight,
                copied=True,
            )
        else:
            keypoints = None

        return AnnotatedThrow(throw_data=self.throw_data, keypoints=keypoints)
