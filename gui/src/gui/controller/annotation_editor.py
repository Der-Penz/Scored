"""Labeling logic shared by the game and the label mode.

The editor owns the state of the frame on screen: the draft tip, the selected
dart and the rules for turning clicks into keypoints. It never touches the
panel itself; the controller that owns it decides when to redraw and what the
frame header says.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime

from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.annotation.keypoints import DartKeypoints
from scored_lib.annotation.throw_annotation import AnnotatedThrow
from scored_lib.dart.constants import Position

from gui.view.annotation.annotation_canvas import AnnotationPoint
from gui.view.drawing.dartboard import DART_COLORS

TEXT_WIDGETS = ("TEntry", "TCombobox", "TText")


def key_targets_annotation(widget: tk.Misc | None, view: tk.Misc) -> bool:
    """Check whether a key event should reach the annotation panel.

    Parameters
    ----------
    widget : tk.Misc | None
        The widget that received the key event.
    view : tk.Misc
        The annotation panel the event has to belong to.

    Returns
    -------
    bool
        True when the panel is on screen and the focus is not in a text widget.
    """
    if widget is None:
        return False
    try:
        if not view.winfo_viewable():
            return False
        if widget.winfo_class() in TEXT_WIDGETS:
            return False
    except (KeyError, TypeError, tk.TclError):
        return False

    current: tk.Misc | None = widget
    while current is not None:
        if current is view:
            return True
        parent = current.master
        if parent is None or parent is current:
            return False
        current = parent
    return False


class AnnotationEditor:
    """Edit the keypoints of the darts on one frame.

    Parameters
    ----------
    allow_new_darts : bool
        True to keep labeling when every dart of the annotation is labeled.
        A new dart without a known score is appended for the next pair of
        clicks. False caps the labels at the darts the frame was recorded for,
        which is what a game needs: the darts in the frame are known.
    """

    def __init__(self, allow_new_darts: bool = False) -> None:
        self.allow_new_darts = allow_new_darts
        self._annotation: ImageAnnotation | None = None
        self._draft_tip: Position | None = None
        self._selected_entry: int | None = None

    @property
    def annotation(self) -> ImageAnnotation | None:
        """The annotation being edited."""
        return self._annotation

    @property
    def selected_entry(self) -> int | None:
        """Index of the selected dart, or None."""
        return self._selected_entry

    @property
    def draft_active(self) -> bool:
        """Whether a tip was placed whose flight is still missing."""
        return self._draft_tip is not None

    def set_annotation(self, annotation: ImageAnnotation | None) -> None:
        """Edit *annotation* from now on, dropping the selection and the draft."""
        self._annotation = annotation
        self.reset()

    def reset(self) -> None:
        """Forget the selection and the unfinished draft."""
        self._draft_tip = None
        self._selected_entry = None

    def place(self, x: float, y: float) -> str | None:
        """Place the tip of a dart, or its flight when a tip is waiting.

        Returns
        -------
        str | None
            A message for the user, or None when the click was applied.
        """
        annotation = self._annotation
        if annotation is None:
            return None
        if not self.allow_new_darts and not any(
            entry.keypoints is None for entry in annotation.throws
        ):
            return "Every dart is already labeled."

        if self._draft_tip is None:
            self._draft_tip = (x, y)
            self._selected_entry = None
            return None

        keypoints = DartKeypoints(tip=self._draft_tip, flight=(x, y))
        self._selected_entry = self._commit(annotation, keypoints)
        self._draft_tip = None
        return None

    def select(self, index: int) -> None:
        """Select the dart at *index*."""
        annotation = self._annotation
        if annotation is None or not 0 <= index < len(annotation.throws):
            return
        self._draft_tip = None
        self._selected_entry = index

    def move(self, index: int, kind: str, x: float, y: float) -> None:
        """Move the tip or the flight of the dart at *index* to ``(x, y)``."""
        annotation = self._annotation
        if annotation is None or not 0 <= index < len(annotation.throws):
            return
        entry = annotation.throws[index]
        if entry.keypoints is None:
            return

        tip = entry.keypoints.tip if kind != "tip" else (x, y)
        flight = entry.keypoints.flight if kind != "flight" else (x, y)
        keypoints = DartKeypoints(tip=tip, flight=flight, copied=entry.keypoints.copied)
        annotation.throws[index] = AnnotatedThrow(entry.throw_data, keypoints)
        self._selected_entry = index

    def delete(self) -> None:
        """Cancel the draft, or clear the label of the selected dart."""
        if self._draft_tip is not None:
            self._draft_tip = None
            self._selected_entry = None
            return

        annotation = self._annotation
        if annotation is None or self._selected_entry is None:
            return
        if not 0 <= self._selected_entry < len(annotation.throws):
            return
        entry = annotation.throws[self._selected_entry]
        annotation.throws[self._selected_entry] = AnnotatedThrow(entry.throw_data, None)
        self._selected_entry = None

    def copy_from(self, source: ImageAnnotation) -> int:
        """Copy the labeled darts of *source* onto matching unlabeled darts.

        A dart only matches when it carries the same throw data, so copied
        labels always stay attached to the score they were made for.

        Returns
        -------
        int
            The number of darts that were labeled.
        """
        annotation = self._annotation
        if annotation is None:
            return 0

        copied = 0
        for previous in source.throws:
            if previous.keypoints is None or self._has_keypoints(previous.keypoints):
                continue
            target = next(
                (
                    index
                    for index, entry in enumerate(annotation.throws)
                    if entry.keypoints is None
                    and previous.throw_data is not None
                    and entry.throw_data == previous.throw_data
                ),
                None,
            )
            if target is None:
                continue
            annotation.throws[target] = previous.make_copy()
            copied += 1
        return copied

    def points(self) -> list[AnnotationPoint]:
        """Return the markers to draw on the frame."""
        annotation = self._annotation
        if annotation is None:
            return []

        points: list[AnnotationPoint] = []
        for index, entry in enumerate(annotation.throws):
            if entry.keypoints is None:
                continue
            selected = index == self._selected_entry
            color = DART_COLORS[index % len(DART_COLORS)]
            points.append(
                AnnotationPoint(
                    entry.keypoints.tip[0],
                    entry.keypoints.tip[1],
                    index=index,
                    kind="tip",
                    label="T",
                    color=color,
                    selected=selected,
                )
            )
            points.append(
                AnnotationPoint(
                    entry.keypoints.flight[0],
                    entry.keypoints.flight[1],
                    index=index,
                    kind="flight",
                    label="F",
                    color=color,
                    selected=selected,
                )
            )
        if self._draft_tip is not None:
            points.append(AnnotationPoint(self._draft_tip[0], self._draft_tip[1], index=-1))
        return points

    def rows(self) -> list[tuple[str, str, str, str, str]]:
        """Return ``(index, score, tip, flight, source)`` for every dart."""
        annotation = self._annotation
        if annotation is None:
            return []

        rows: list[tuple[str, str, str, str, str]] = []
        for index, entry in enumerate(annotation.throws):
            score = entry.throw_data.short_label if entry.throw_data is not None else "—"
            if entry.keypoints is None:
                rows.append((str(index + 1), score, "—", "—", "—"))
                continue
            source = "copied" if entry.keypoints.copied else "manual"
            rows.append(
                (
                    str(index + 1),
                    score,
                    _format_point(entry.keypoints.tip),
                    _format_point(entry.keypoints.flight),
                    source,
                )
            )
        return rows

    def summary(self) -> str:
        """Describe how much of the frame is labeled, e.g. ``2/3 darts labeled``."""
        annotation = self._annotation
        if annotation is None:
            return ""
        labeled = sum(entry.keypoints is not None for entry in annotation.throws)
        total = len(annotation.throws)
        plural = "" if total == 1 else "s"
        bust = " · bust" if annotation.is_bust else ""
        return f"{labeled}/{total} dart{plural} labeled{bust}"

    def frame_header(self) -> str:
        """Describe the recorded throw of the annotation."""
        annotation = self._annotation
        if annotation is None:
            return ""
        timestamp = datetime.fromtimestamp(annotation.timestamp).strftime("%H:%M:%S")
        return (
            f"Round {annotation.round} · Throw {annotation.throw} · {timestamp} · {self.summary()}"
        )

    def _commit(self, annotation: ImageAnnotation, keypoints: DartKeypoints) -> int | None:
        """Attach *keypoints* to the first dart without keypoints.

        Returns
        -------
        int | None
            The index of the dart the keypoints were attached to, or None when
            every dart already has keypoints and no new dart may be added.
        """
        for index, entry in enumerate(annotation.throws):
            if entry.keypoints is None:
                annotation.throws[index] = AnnotatedThrow(entry.throw_data, keypoints)
                return index
        if not self.allow_new_darts:
            return None
        annotation.throws.append(AnnotatedThrow(None, keypoints))
        return len(annotation.throws) - 1

    def _has_keypoints(self, keypoints: DartKeypoints) -> bool:
        annotation = self._annotation
        if annotation is None:
            return False
        return any(
            entry.keypoints is not None
            and entry.keypoints.tip == keypoints.tip
            and entry.keypoints.flight == keypoints.flight
            for entry in annotation.throws
        )


def _format_point(point: Position) -> str:
    """Format a keypoint for the table."""
    return f"({point[0]:.2f}, {point[1]:.2f})"
