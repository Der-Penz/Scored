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
    throw_data : DartThrow | None
        The score the dart scored, None for darts that carry no known score,
        e.g. a leftover dart of an earlier visit or a dart labeled on a
        standalone image that was never part of a game.
    keypoints : DartKeypoints | None
        The labeled tip and flight of this dart, by default unlabeled.
    """

    throw_data: DartThrow | None
    keypoints: DartKeypoints | None = None

    def make_copy(self) -> AnnotatedThrow:
        """Make a copy of this AnnotatedThrow."""

        if self.keypoints is not None:
            keypoints = DartKeypoints(
                tip=self.keypoints.tip,
                flight=self.keypoints.flight,
                copied=True,
                tip_visible=self.keypoints.tip_visible,
                flight_visible=self.keypoints.flight_visible,
            )
        else:
            keypoints = None

        return AnnotatedThrow(throw_data=self.throw_data, keypoints=keypoints)
