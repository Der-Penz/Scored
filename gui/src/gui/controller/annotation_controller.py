from __future__ import annotations

import logging
import tkinter as tk

import ttkbootstrap as ttk
from scored_lib.annotation.leg_annotation import LegAnnotation
from scored_lib.annotation.leg_annotation_handler import LegAnnotationHandler

import gui.events.event_types as events
from gui.controller.annotation_editor import AnnotationEditor, key_targets_annotation
from gui.events.event_channel import EventChannel
from gui.model.args import AppMode
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.annotation.annotation_view import AnnotationView
from gui.view.app_view import AppView


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
    shows a banner saying so. In label mode this controller stays idle and
    :class:`~gui.controller.label_controller.LabelController` owns the panel.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self.annotation_handlers: dict[str, LegAnnotationHandler] = {}
        self.editor = AnnotationEditor(allow_new_darts=False)

        self._selected_throw: int | None = None
        self._follow_newest = True
        self._keys_bound = False

    @property
    def annotation_view(self) -> AnnotationView:
        return self._view.annotation_view

    @property
    def _active(self) -> bool:
        """Whether the game owns the annotation panel right now."""
        return self._model.mode is AppMode.GAME

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
        self._event_channel.subscribe(events.ModeChanged, self._on_mode_changed)
        self._refresh()

    def _on_key(self, event: tk.Event) -> str | None:
        if not self._active:
            return None
        if event.state & 0x0004:
            return None
        key = (getattr(event, "keysym", "") or "").lower()
        if not key_targets_annotation(event.widget, self.annotation_view):
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

    def _on_throw_selected(self, throw: int) -> None:
        if not self._active:
            return
        if throw not in (1, 2, 3):
            self._selected_throw = None
        else:
            self._selected_throw = throw - 1
        self._follow_newest = False
        self.editor.reset()
        self._refresh()

    def _select_throw(self, throw: int) -> None:
        """Switch to the tab of *throw* if the round has a sample for it."""
        if throw <= len(self._model.annotations):
            self._on_throw_selected(throw)

    def _select_dart(self, index: int) -> None:
        """Select the dart at *index* (0, 1 or 2) in the current frame."""
        self._on_point_selected(index, None)

    def _on_point_placed(self, x: float, y: float) -> None:
        if not self._active:
            return
        message = self.editor.place(x, y)
        self._refresh()
        if message is not None:
            self.annotation_view.set_status(message)

    def _on_point_selected(self, index: int) -> None:
        if not self._active:
            return
        self.editor.select(index)
        self._refresh()

    def _on_point_moved(self, index: int, kind: str, x: float, y: float) -> None:
        if not self._active:
            return
        self.editor.move(index, kind, x, y)
        self._refresh()

    def _on_copy(self) -> None:
        if not self._active:
            return
        annotation = self.editor.annotation
        if annotation is None or self._selected_throw is None:
            return
        if self._selected_throw == 0:
            self.annotation_view.set_status("Nothing to copy from the first throw.")
            return

        previous, _ = self._model.annotations[self._selected_throw - 1]
        copied = self.editor.copy_from(previous)
        message = (
            "Already copied."
            if copied == 0
            else f"Copied {copied} dart{'s' if copied != 1 else ''}."
        )
        self._refresh()
        self.annotation_view.set_status(message)

    def _on_delete(self) -> None:
        if not self._active:
            return
        self.editor.delete()
        self._refresh()

    def _refresh(self) -> None:
        view = self.annotation_view
        view.set_toolbar_mode("round")
        self._apply_data_notice()
        annotations = self._model.annotations
        available = [annotation.throw for annotation, _ in annotations]
        view.set_throw_tabs(available)

        if not annotations:
            view.clear()
            self._selected_throw = None
            self.editor.reset()
            return

        newest = len(annotations) - 1
        if self._selected_throw is None or self._selected_throw > newest or self._follow_newest:
            self._selected_throw = newest

        annotation, image = annotations[self._selected_throw]
        if self.editor.annotation is not annotation:
            self.editor.set_annotation(annotation)
        view.show_throw(
            throw=annotation.throw,
            image=image,
            points=self.editor.points(),
            rows=self.editor.rows(),
            selected=self.editor.selected_entry,
            metrics=self.editor.frame_header(),
        )

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
        if not self._active:
            return
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
            if self._active:
                self._refresh()

    def _on_mode_changed(self, event: events.ModeChanged) -> None:
        if event.mode is AppMode.GAME:
            self.bind_components()
            self._refresh()
            return
        self.save_round()
        self.reset_selection()

    def reset_selection(self) -> None:
        self._selected_throw = None
        self.editor.reset()
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
