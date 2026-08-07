from functools import lru_cache

from scored_lib.dart.dart_throw import DartThrow
from scored_lib.dart.multiplier import Multiplier
from scored_lib.game.rule import FinishRule


def _checkout_path_key(
    path: tuple[DartThrow, ...],
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        len(path),
        tuple(-dart_throw.score for dart_throw in path),
        tuple(-dart_throw.multiplier.value for dart_throw in path),
        tuple(-dart_throw.number for dart_throw in path),
    )


def _direct_checkout_paths(score: int) -> tuple[tuple[DartThrow, ...], ...]:
    candidates: list[tuple[DartThrow, ...]] = []

    if 1 <= score <= 20:
        candidates.append((DartThrow(score, Multiplier.SINGLE),))
    if score == 25:
        candidates.append((DartThrow(0, Multiplier.OUTER_BULL),))
    if score == 50:
        candidates.append((DartThrow(0, Multiplier.INNER_BULL),))
    if 2 <= score <= 40 and score % 2 == 0:
        candidates.append((DartThrow(score // 2, Multiplier.DOUBLE),))
    if 3 <= score <= 60 and score % 3 == 0:
        candidates.append((DartThrow(score // 3, Multiplier.TRIPLE),))

    return tuple(candidates)


@lru_cache(maxsize=None)
def get_best_checkout_path(
    score: int, finish_rule: FinishRule
) -> tuple[DartThrow, ...] | None:
    if score <= 0:
        return None

    if finish_rule == FinishRule.ANY:
        direct_paths = _direct_checkout_paths(score)
        if direct_paths:
            return min(direct_paths, key=_checkout_path_key)
        finish_rule = FinishRule.DOUBLE

    if finish_rule in {FinishRule.DOUBLE, FinishRule.DOUBLE_OR_BULL}:
        paths = FINISHES.get(str(score))
        if not paths:
            return None
        return min((tuple(path) for path in paths), key=_checkout_path_key)

    if finish_rule == FinishRule.SINGLE:
        if score > 60:
            return None

        path: list[DartThrow] = []
        remaining = score
        while remaining > 20:
            path.append(DartThrow(20, Multiplier.SINGLE))
            remaining -= 20
        path.append(DartThrow(remaining, Multiplier.SINGLE))
        return tuple(path)

    if finish_rule == FinishRule.TRIPLE:
        if 3 <= score <= 60 and score % 3 == 0:
            return (DartThrow(score // 3, Multiplier.TRIPLE),)
        return None

    raise ValueError(f"Unsupported finish rule: {finish_rule}")


FINISHES = {
    "41": [
        [DartThrow(1, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(9, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "42": [
        [DartThrow(2, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(10, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "43": [
        [DartThrow(3, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(11, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "44": [
        [DartThrow(4, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(12, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "45": [
        [DartThrow(5, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(13, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "46": [
        [DartThrow(6, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(14, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "47": [
        [DartThrow(7, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(15, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "48": [
        [DartThrow(8, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(16, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "49": [
        [DartThrow(9, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(17, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "50": [
        [DartThrow(10, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(18, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "51": [
        [DartThrow(11, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(19, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "52": [
        [DartThrow(12, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(20, Multiplier.SINGLE), DartThrow(16, Multiplier.DOUBLE)],
    ],
    "53": [
        [DartThrow(13, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(17, Multiplier.SINGLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "54": [
        [DartThrow(14, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(18, Multiplier.SINGLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "55": [
        [DartThrow(15, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(19, Multiplier.SINGLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "56": [
        [DartThrow(16, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(20, Multiplier.SINGLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "57": [[DartThrow(17, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)]],
    "58": [[DartThrow(18, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)]],
    "59": [[DartThrow(19, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)]],
    "60": [[DartThrow(20, Multiplier.SINGLE), DartThrow(20, Multiplier.DOUBLE)]],
    "61": [
        [DartThrow(15, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)],
        [DartThrow(11, Multiplier.TRIPLE), DartThrow(14, Multiplier.DOUBLE)],
    ],
    "62": [
        [DartThrow(10, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)],
        [DartThrow(12, Multiplier.TRIPLE), DartThrow(13, Multiplier.DOUBLE)],
    ],
    "63": [
        [DartThrow(13, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)],
        [DartThrow(17, Multiplier.TRIPLE), DartThrow(6, Multiplier.DOUBLE)],
    ],
    "64": [
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)],
        [DartThrow(14, Multiplier.TRIPLE), DartThrow(11, Multiplier.DOUBLE)],
    ],
    "65": [
        [DartThrow(11, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)],
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(4, Multiplier.DOUBLE)],
        [DartThrow(15, Multiplier.TRIPLE), DartThrow(10, Multiplier.DOUBLE)],
    ],
    "66": [
        [DartThrow(10, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)],
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(6, Multiplier.DOUBLE)],
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(9, Multiplier.DOUBLE)],
    ],
    "67": [
        [DartThrow(9, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(17, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)],
    ],
    "68": [
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(10, Multiplier.DOUBLE)],
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(7, Multiplier.DOUBLE)],
    ],
    "69": [[DartThrow(19, Multiplier.TRIPLE), DartThrow(6, Multiplier.DOUBLE)]],
    "70": [
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)],
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(5, Multiplier.DOUBLE)],
    ],
    "71": [
        [DartThrow(13, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)],
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(7, Multiplier.DOUBLE)],
    ],
    "72": [
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)],
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(6, Multiplier.DOUBLE)],
    ],
    "73": [[DartThrow(19, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)]],
    "74": [
        [DartThrow(14, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)],
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(13, Multiplier.DOUBLE)],
    ],
    "75": [[DartThrow(17, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)]],
    "76": [
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(8, Multiplier.DOUBLE)],
        [DartThrow(16, Multiplier.TRIPLE), DartThrow(14, Multiplier.DOUBLE)],
    ],
    "77": [[DartThrow(19, Multiplier.TRIPLE), DartThrow(10, Multiplier.DOUBLE)]],
    "78": [[DartThrow(18, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)]],
    "79": [
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(11, Multiplier.DOUBLE)],
        [DartThrow(13, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
    ],
    "80": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(10, Multiplier.DOUBLE)]],
    "81": [
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)],
        [DartThrow(15, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "82": [
        [DartThrow(0, Multiplier.INNER_BULL), DartThrow(16, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(17, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [DartThrow(14, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
    ],
    "83": [[DartThrow(17, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)]],
    "84": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(12, Multiplier.DOUBLE)]],
    "85": [
        [DartThrow(15, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(14, Multiplier.DOUBLE)],
    ],
    "86": [[DartThrow(18, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)]],
    "87": [[DartThrow(17, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)]],
    "88": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(14, Multiplier.DOUBLE)]],
    "89": [[DartThrow(19, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)]],
    "90": [
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(15, Multiplier.DOUBLE)],
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)],
    ],
    "91": [
        [DartThrow(17, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(16, Multiplier.TRIPLE),
            DartThrow(9, Multiplier.DOUBLE),
        ],
    ],
    "92": [
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(16, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ],
    ],
    "93": [
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(7, Multiplier.DOUBLE),
        ],
    ],
    "94": [
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.DOUBLE),
        ],
    ],
    "95": [
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(19, Multiplier.DOUBLE)],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(5, Multiplier.DOUBLE),
        ],
    ],
    "96": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(18, Multiplier.DOUBLE)]],
    "97": [[DartThrow(19, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)]],
    "98": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(19, Multiplier.DOUBLE)]],
    "99": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.SINGLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
    ],
    "100": [[DartThrow(20, Multiplier.TRIPLE), DartThrow(20, Multiplier.DOUBLE)]],
    "101": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(9, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [DartThrow(17, Multiplier.TRIPLE), DartThrow(0, Multiplier.INNER_BULL)],
    ],
    "102": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.SINGLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
    ],
    "103": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.SINGLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
    ],
    "104": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(15, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [DartThrow(18, Multiplier.TRIPLE), DartThrow(0, Multiplier.INNER_BULL)],
    ],
    "105": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(13, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ]
    ],
    "106": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ]
    ],
    "107": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.DOUBLE),
        ],
        [DartThrow(19, Multiplier.TRIPLE), DartThrow(0, Multiplier.INNER_BULL)],
    ],
    "108": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.SINGLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "109": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(9, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "110": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.DOUBLE),
        ],
        [DartThrow(20, Multiplier.TRIPLE), DartThrow(0, Multiplier.INNER_BULL)],
    ],
    "111": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(11, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "112": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ]
    ],
    "113": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "114": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "115": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(15, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "116": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "117": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "118": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "119": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.TRIPLE),
            DartThrow(13, Multiplier.DOUBLE),
        ]
    ],
    "120": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.SINGLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "121": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(11, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.DOUBLE),
        ],
        [
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(5, Multiplier.DOUBLE),
        ],
    ],
    "122": [
        [
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(7, Multiplier.DOUBLE),
        ]
    ],
    "123": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.TRIPLE),
            DartThrow(9, Multiplier.DOUBLE),
        ]
    ],
    "124": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.TRIPLE),
            DartThrow(11, Multiplier.DOUBLE),
        ]
    ],
    "125": [
        [
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(7, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(15, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.DOUBLE),
        ],
    ],
    "126": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.DOUBLE),
        ]
    ],
    "127": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ]
    ],
    "128": [
        [
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(7, Multiplier.DOUBLE),
        ],
    ],
    "129": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(6, Multiplier.DOUBLE),
        ],
    ],
    "130": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(5, Multiplier.DOUBLE),
        ]
    ],
    "131": [
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(13, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
    ],
    "132": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ],
    ],
    "133": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ]
    ],
    "134": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.TRIPLE),
            DartThrow(13, Multiplier.DOUBLE),
        ]
    ],
    "135": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ],
        [
            DartThrow(0, Multiplier.OUTER_BULL),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ],
    ],
    "136": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(8, Multiplier.DOUBLE),
        ]
    ],
    "137": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.DOUBLE),
        ]
    ],
    "138": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ],
    ],
    "139": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(13, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(11, Multiplier.DOUBLE),
        ],
    ],
    "140": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(10, Multiplier.DOUBLE),
        ]
    ],
    "141": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ]
    ],
    "142": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.DOUBLE),
        ],
    ],
    "143": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
    ],
    "144": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(12, Multiplier.DOUBLE),
        ]
    ],
    "145": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.DOUBLE),
        ]
    ],
    "146": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ],
    ],
    "147": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
    ],
    "148": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(14, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "149": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ]
    ],
    "150": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ],
    ],
    "151": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ],
    ],
    "152": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(16, Multiplier.DOUBLE),
        ]
    ],
    "153": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ]
    ],
    "154": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "155": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.DOUBLE),
        ]
    ],
    "156": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.DOUBLE),
        ]
    ],
    "157": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "158": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.DOUBLE),
        ]
    ],
    "160": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.DOUBLE),
        ]
    ],
    "161": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(17, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ]
    ],
    "164": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(18, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ],
        [
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ],
    ],
    "167": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(19, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ]
    ],
    "170": [
        [
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(20, Multiplier.TRIPLE),
            DartThrow(0, Multiplier.INNER_BULL),
        ]
    ],
}
