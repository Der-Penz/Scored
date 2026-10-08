from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin, dataclass_json

from scored_lib.dart.constants import Position


@dataclass_json
@dataclass(frozen=True, eq=True)
class DartKeypoints(DataClassJsonMixin):
    """The two labeled keypoints of one dart.

    Parameters
    ----------
    tip : Position
        Relative image space coordinates of the dart tip.
    flight : Position
        Relative image space coordinates of the flight tip.
    copied : bool
        True when the points were copied from the previous throw's image
        instead of being placed by hand.
    """

    tip: Position
    flight: Position
    copied: bool = False
