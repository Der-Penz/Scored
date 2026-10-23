from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, config, dataclass_json

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
    tip_visible : bool
        True when the tip is actually visible in the image. Invisible points
        stay labeled but are not used as ground truth keypoints.
    flight_visible : bool
        True when the flight is actually visible in the image.
    """

    tip: Position = field(metadata=config(encoder=list, decoder=tuple))
    flight: Position = field(metadata=config(encoder=list, decoder=tuple))
    copied: bool = False
    tip_visible: bool = True
    flight_visible: bool = True
