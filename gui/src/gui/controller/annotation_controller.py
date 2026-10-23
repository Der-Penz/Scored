from __future__ import annotations

import logging
import tkinter as tk
from datetime import datetime

import ttkbootstrap as ttk
from scored_lib.annotation.keypoints import DartKeypoints
from scored_lib.annotation.leg_annotation import LegAnnotation
from scored_lib.annotation.leg_annotation_handler import LegAnnotationHandler
from scored_lib.annotation.throw_annotation import AnnotatedThrow
from scored_lib.dart.constants import Position

import gui.events.event_types as events
from gui.events.event_channel import EventChannel
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.annotation.annotation_canvas import AnnotationPoint
from gui.view.annotation.annotation_view import AnnotationView
from gui.view.app_view import AppView
from gui.view.drawing.dartboard import DART_COLORS


class AnnotationController(BaseController):
    """Annotate the keypoints of the darts in the captured camera frames.

    The panel labels the throws of the current round against the frames that
    were collected while it was played: one image per throw, showing the darts
    that are visible on the board. A click places the tip first, the second
    click the flight, and the finished object is attached to the first dart
    that has no keypoints yet; an annotation never grows past the darts in the
    frame. ``V`` copies the labels of the previous throw, the Delete key
    clears the label of the selected dart (keeping the dart with its score) or
    cancels an unfinished draft, ``1``/``2``/``3`` switch the throw tabs and
    ``A``/``S``/``D`` select the first, second or third dart. Selecting a point
    on the canvas and pressing ``H`` toggles whether that keypoint is visible;
    without a point selected, ``H`` toggles both keypoints of the selected
    dart. Hidden keypoints are drawn hollow and dashed, and are left out of
    the ground truth keypoints.

    The labels stay in the model while the round is playing and are written to
    disk once the round is over: when the turn is confirmed, when the game
    ends, or when the app is closed. While data collection is disabled (no
    directory is set) no frames or labels are collected at all, and the panel
    shows a banner saying so.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self.annotation_handlers: dict[str, LegAnnotationHandler] = {}

        self._selected_throw: int | None = None
        self._selected_entry: int | None = None
        self._selected_kind: str | None = None
        self._draft_tip: Position | None = None
        self._follow_newest = True
        self._keys_bound = False

    @property
    def annotation_view(self) -> AnnotationView:
        return self._view.annotation_view

    def bind_menu(self, _: ttk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        view = self.annotation_view
        view.on_throw_selected = self._on_throw_selected
        view.on_copy_clicked = self._on_copy
        view.on_delete_clicked = self._on_delete
        view.on_point_placed = self._on_point_placed
        view.on_point_selected = self._on_point_selected
        view.on_point_moved = self._on_point_moved
        if not self._keys_bound:
            self._keys_bound = True
            self._view.master.bind_all("<Key>", self._on_key)

    def start(self) -> None:
        self._event_channel.subscribe(events.TurnChanged, self._on_turn_changed)
        self._event_channel.subscribe(events.GameStarted, self._on_game_started)
        self._event_channel.subscribe(events.GameStateChanged, self._on_game_state_changed)
        self._event_channel.subscribe(events.GameEnded, self._on_game_ended)
        self._event_channel.subscribe(events.PanelRebound, self._on_panel_rebound)
        self._event_channel.subscribe(
            events.DataCollectionChanged, self._on_data_collection_changed
        )
        self._refresh()

    def _on_key(self, event: tk.Event) -> str | None:
        if event.state & 0x0004:
            return None
        key = (getattr(event, "keysym", "") or "").lower()
        if not self._is_annotation_widget(event.widget):
            return None
        if key in ("1", "2", "3"):
            self._select_throw(int(key))
        elif key in ("a", "s", "d"):
            self._select_dart({"a": 0, "s": 1, "d": 2}[key])
        elif key == "v":
            self._on_copy()
        elif key == "h":
            self._toggle_visibility()
        elif key in ("delete", "kp_delete"):
            self._on_delete()
        else:
            return None
        return "break"

    def _is_annotation_widget(self, widget: tk.Misc | None) -> bool:
        """Check whether a key event should reach the annotation panel."""
        if widget is None:
            return False
        try:
            view = self.annotation_view
            if not view.winfo_viewable():
                return False
            if widget.winfo_class() in ("TEntry", "TCombobox", "TText"):
                return False
        except (KeyError, TypeError, tk.TclError):
            return False

        current = widget
        while True:
            if current is view:
                return True
            parent = current.master
            if parent is None or parent is current:
                return False
            current = parent

    def _on_throw_selected(self, throw: int) -> None:
        if throw not in (1, 2, 3):
            self._selected_throw = None
        else:
            self._selected_throw = throw - 1
        self._follow_newest = False
        self._selected_entry = None
        self._selected_kind = None
        self._draft_tip = None
        self._refresh()

    def _select_throw(self, throw: int) -> None:
        """Switch to the tab of *throw* if the round has a sample for it."""
        if throw <= len(self._model.annotations):
            self._on_throw_selected(throw)

    def _select_dart(self, index: int) -> None:
        """Select the dart at *index* (0, 1 or 2) in the current frame."""
        self._on_point_selected(index, None)

    def _on_point_placed(self, x: float, y: float) -> None:
        annotation = self._current_annotation()
        if annotation is None:
            return
        if not any(entry.keypoints is None for entry in annotation.throws):
            self.annotation_view.set_status("Every dart is already labeled.")
            self._refresh()
            return
        if self._draft_tip is None:
            self._draft_tip = (x, y)
            self._selected_entry = None
            self._selected_kind = None
        else:
            keypoints = DartKeypoints(tip=self._draft_tip, flight=(x, y))
            self._selected_entry = self._commit_keypoints(annotation, keypoints)
            self._selected_kind = None
            self._draft_tip = None
        self._refresh()

    def _on_point_selected(self, index: int, kind: str | None = None) -> None:
        annotation = self._current_annotation()
        if annotation is None:
            return
        if not 0 <= index < len(annotation.throws):
            return
        self._draft_tip = None
        self._selected_entry = index
        self._selected_kind = kind if kind in ("tip", "flight") else None
        self._refresh()

    def _on_point_moved(self, index: int, kind: str, x: float, y: float) -> None:
        annotation = self._current_annotation()
        if annotation is None or not 0 <= index < len(annotation.throws):
            return
        entry = annotation.throws[index]
        if entry.keypoints is None:
            return

        tip = entry.keypoints.tip if kind != "tip" else (x, y)
        flight = entry.keypoints.flight if kind != "flight" else (x, y)
        keypoints = DartKeypoints(
            tip=tip,
            flight=flight,
            copied=entry.keypoints.copied,
            tip_visible=entry.keypoints.tip_visible,
            flight_visible=entry.keypoints.flight_visible,
        )
        annotation.throws[index] = AnnotatedThrow(entry.throw_data, keypoints)
        self._selected_entry = index
        self._selected_kind = kind if kind in ("tip", "flight") else None
        self._refresh()

    def _commit_keypoints(self, annotation, keypoints: DartKeypoints) -> int | None:
        """Attach *keypoints* to the first dart without keypoints.

        The annotation never grows past the darts that are visible in the
        frame, so a fully labeled annotation is left untouched.

        Returns
        -------
        int | None
            The index of the dart the keypoints were attached to, or None when
            every dart already has keypoints.
        """
        for index, entry in enumerate(annotation.throws):
            if entry.keypoints is None:
                annotation.throws[index] = AnnotatedThrow(entry.throw_data, keypoints)
                return index
        return None

    def _on_copy(self) -> None:
        annotation = self._current_annotation()
        if annotation is None:
            return
        if self._selected_throw == 0:
            self.annotation_view.set_status("Nothing to copy from the first throw.")
            return

        previous, _ = self._model.annotations[self._selected_throw - 1]
        copied = 0
        for source in previous.throws:
            if source.keypoints is None:
                continue
            if self._has_keypoints(annotation, source.keypoints):
                continue
            target_index = next(
                (
                    index
                    for index, target in enumerate(annotation.throws)
                    if target.keypoints is None
                    and source.throw_data is not None
                    and target.throw_data == source.throw_data
                ),
                None,
            )
            if target_index is None:
                continue
            annotation.throws[target_index] = source.make_copy()
            copied += 1

        message = (
            "Already copied."
            if copied == 0
            else f"Copied {copied} dart{'s' if copied != 1 else ''}."
        )
        self._refresh()
        self.annotation_view.set_status(message)

    def _has_keypoints(self, annotation, keypoints: DartKeypoints) -> bool:
        return any(
            entry.keypoints is not None
            and entry.keypoints.tip == keypoints.tip
            and entry.keypoints.flight == keypoints.flight
            for entry in annotation.throws
        )

    def _toggle_visibility(self) -> None:
        """Toggle the visibility of the selected keypoint or dart."""
        annotation = self._current_annotation()
        if annotation is None or self._selected_entry is None:
            return
        if not 0 <= self._selected_entry < len(annotation.throws):
            return
        entry = annotation.throws[self._selected_entry]
        keypoints = entry.keypoints
        if keypoints is None:
            return

        tip_visible = keypoints.tip_visible
        flight_visible = keypoints.flight_visible
        if self._selected_kind in (None, "tip"):
            tip_visible = not tip_visible
        if self._selected_kind in (None, "flight"):
            flight_visible = not flight_visible

        annotation.throws[self._selected_entry] = AnnotatedThrow(
            entry.throw_data,
            DartKeypoints(
                tip=keypoints.tip,
                flight=keypoints.flight,
                copied=keypoints.copied,
                tip_visible=tip_visible,
                flight_visible=flight_visible,
            ),
        )
        if self._selected_kind is None:
            message = "Keypoints visible." if tip_visible or flight_visible else "Keypoints hidden."
        else:
            visible = tip_visible if self._selected_kind == "tip" else flight_visible
            name = "Tip" if self._selected_kind == "tip" else "Flight"
            message = f"{name} visible." if visible else f"{name} hidden."
        self._refresh()
        self.annotation_view.set_status(message)

    def _on_delete(self) -> None:
        if self._draft_tip is not None:
            self._draft_tip = None
            self._selected_entry = None
            self._selected_kind = None
            self._refresh()
            return

        annotation = self._current_annotation()
        if annotation is None or self._selected_entry is None:
            return
        if not 0 <= self._selected_entry < len(annotation.throws):
            return
        entry = annotation.throws[self._selected_entry]
        annotation.throws[self._selected_entry] = AnnotatedThrow(entry.throw_data, None)
        self._selected_entry = None
        self._selected_kind = None
        self._refresh()

    def _refresh(self) -> None:
        view = self.annotation_view
        self._apply_data_notice()
        annotations = self._model.annotations
        available = [annotation.throw for annotation, _ in annotations]
        view.set_throw_tabs(available)

        if not annotations:
            view.clear()
            self._selected_throw = None
            self._selected_entry = None
            self._selected_kind = None
            self._draft_tip = None
            return

        newest = len(annotations) - 1
        if self._selected_throw is None or self._selected_throw > newest or self._follow_newest:
            self._selected_throw = newest

        annotation, image = annotations[self._selected_throw]
        view.show_throw(
            throw=annotation.throw,
            image=image,
            points=self._points_for(annotation),
            rows=self._rows_for(annotation),
            selected=self._selected_entry,
            metrics=self._metrics_for(annotation),
        )

    def _current_annotation(self):
        if self._selected_throw is None:
            return None
        annotations = self._model.annotations
        if not 0 <= self._selected_throw < len(annotations):
            return None
        return annotations[self._selected_throw][0]

    def _points_for(self, annotation) -> list[AnnotationPoint]:
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
                    selected=selected and self._selected_kind in (None, "tip"),
                    visible=entry.keypoints.tip_visible,
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
                    selected=selected and self._selected_kind in (None, "flight"),
                    visible=entry.keypoints.flight_visible,
                )
            )
        if self._draft_tip is not None:
            points.append(AnnotationPoint(self._draft_tip[0], self._draft_tip[1], index=-1))
        return points

    def _rows_for(self, annotation) -> list[tuple[str, str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str, str]] = []
        for index, entry in enumerate(annotation.throws):
            score = entry.throw_data.short_label if entry.throw_data is not None else "—"
            if entry.keypoints is None:
                rows.append((str(index + 1), score, "—", "—", "—", "—"))
            else:
                source = "copied" if entry.keypoints.copied else "manual"
                rows.append(
                    (
                        str(index + 1),
                        score,
                        self._format_point(entry.keypoints.tip),
                        self._format_point(entry.keypoints.flight),
                        source,
                        self._visibility_label(entry.keypoints),
                    )
                )
        return rows

    def _visibility_label(self, keypoints: DartKeypoints) -> str:
        """List which keypoints of a dart are visible, e.g. ``"T, F"``."""
        visible = [
            name
            for name, flag in (
                ("T", keypoints.tip_visible),
                ("F", keypoints.flight_visible),
            )
            if flag
        ]
        return ", ".join(visible) if visible else "none"

    def _metrics_for(self, annotation) -> str:
        timestamp = datetime.fromtimestamp(annotation.timestamp).strftime("%H:%M:%S")
        bust = " · bust" if annotation.is_bust else ""
        labeled = sum(entry.keypoints is not None for entry in annotation.throws)
        total = len(annotation.throws)
        plural = "" if total == 1 else "s"
        return (
            f"Round {annotation.round} · Throw {annotation.throw} · "
            f"{timestamp} · {labeled}/{total} dart{plural} labeled{bust}"
        )

    def _format_point(self, point: Position) -> str:
        return f"({point[0]:.2f}, {point[1]:.2f})"

    def _on_game_started(self, _: events.GameStarted) -> None:
        self.annotation_view.clear()
        self._model.annotations.clear()
        self.reset_selection()

        if self._model.game is None:
            return

        if not self._model.data_collection_enabled:
            logging.info("Data collection is disabled, no annotations will be saved.")
            return

        self._create_handlers()

    def _create_handlers(self) -> None:
        """Create one annotation handler per player in the current data directory."""
        directory = self._model.data_dir
        if directory is None or self._model.game is None:
            return

        self.annotation_handlers.clear()
        for player in self._model.players:
            leg_info = LegAnnotation(
                leg_id=f"{player.name}_{player.id}",
                player_name=player.name,
                starting_score=self._model.game.starting_score,
            )
            self.annotation_handlers[player.id] = LegAnnotationHandler(
                directory=directory / player.id, info=leg_info
            )

    def _on_data_collection_changed(self, event: events.DataCollectionChanged) -> None:
        """Follow the data collection toggle from the Data menu."""
        if event.enabled:
            logging.info(f"Data collection enabled, saving data to {event.directory}")
            if self._model.game is not None:
                self._create_handlers()
        else:
            logging.info("Data collection disabled, discarding collected annotations.")
            self.annotation_handlers.clear()
            self._model.annotations.clear()
            self.reset_selection()
        self._refresh()

    def _apply_data_notice(self) -> None:
        """Tell the panel whether the labels it shows are being saved."""
        directory = self._model.data_dir
        self.annotation_view.set_data_collection(enabled=directory is not None, directory=directory)

    def _on_turn_changed(self, _: events.TurnChanged) -> None:
        if self._model.game is None:
            return
        self.save_round()
        self._refresh()

    def _on_game_ended(self, _: events.GameEnded) -> None:
        self.save_round()
        self._refresh()

    def _on_game_state_changed(self, _: events.GameStateChanged) -> None:
        self._refresh()

    def _on_panel_rebound(self, event: events.PanelRebound) -> None:
        if event.panel is Panel.ANNOTATION:
            self.bind_components()
            self._refresh()

    def reset_selection(self) -> None:
        self._selected_throw = None
        self._selected_entry = None
        self._selected_kind = None
        self._draft_tip = None
        self._follow_newest = True

    def save_round(self) -> None:
        """Write the collected round annotations and reset the panel."""
        model = self._model
        if model.game is None or not model.annotations:
            return

        if not model.data_collection_enabled:
            logging.warning(
                f"Data collection is disabled, round for {model.game.last_player.id} not saved."
            )
            model.annotations.clear()
            self.reset_selection()
            return

        first_throw = model.annotations[0][0].throw
        if first_throw != 1:
            # Collection was enabled in the middle of a round, so the first
            # throws of this round are missing and the samples would be out of
            # order. Drop the partial round; the next one is collected whole.
            logging.warning(
                f"Skipping incomplete round for {model.game.last_player.id}: "
                f"collection started at throw {first_throw}."
            )
            model.annotations.clear()
            self.reset_selection()
            return

        player = model.game.last_player
        handler = self.annotation_handlers.get(player.id)
        if handler is None:
            logging.warning(f"No annotation handler for player {player.id}, round not saved.")
            return

        annotations = [annotation for annotation, _ in model.annotations]
        images = [image for _, image in model.annotations]
        try:
            handler.save_round(annotations, images)
        except (OSError, ValueError) as error:
            logging.error(f"Could not save the round annotation: {error}")

        model.annotations.clear()
        self.reset_selection()
