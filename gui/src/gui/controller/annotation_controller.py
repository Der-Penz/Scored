import logging
from pathlib import Path

import ttkbootstrap as ttk
from scored_lib.annotation.leg_annotation import LegAnnotation
from scored_lib.annotation.leg_annotation_handler import LegAnnotationHandler

import gui.events.event_types as events
from gui.events.event_channel import EventChannel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.annotation.annotation_view import AnnotationView
from gui.view.app_view import AppView


class AnnotationController(BaseController):
    """
    Annotate the keypoints of the darts in a camera frame.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self.annotation_handlers: dict[str, LegAnnotationHandler] = {}

        self._output_dir: Path | None = (
            Path(model.config.data_dir) if model.config.data_dir else None
        )

        if self._output_dir:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Collecting data in {self._output_dir}")
            except OSError as error:
                logging.error(
                    f"Could not create the data collection directory: {error}"
                )
                ttk.Messagebox.show_error(
                    message="Could not create the data collection directory.",
                    title="Data Collection Error",
                    parent=self._view,
                )
                self._output_dir = None

    @property
    def annotation_view(self) -> AnnotationView:
        return self._view.annotation_view

    def bind_menu(self, _: ttk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        self._event_channel.subscribe(events.TurnChanged, self._on_turn_changed)
        self._event_channel.subscribe(events.GameStarted, self._on_game_started)
        self._event_channel.subscribe(
            events.GameStateChanged, self._on_game_state_changed
        )

    def _on_turn_changed(self, event: events.TurnChanged) -> None:
        if self._model.game is None:
            return

        self.annotation_view.clear()

        # save the annotation from the model and reset
        if len(self._model.annotations) < 0:
            return

        try:
            annotation_handler = self.annotation_handlers[
                self._model.game.last_player.id
            ]
            annotations = [ann for ann, _ in self._model.annotations]
            images = [img for _, img in self._model.annotations]
            annotation_handler.save_round(annotations, images)
        except KeyError as error:
            logging.error(
                f"Could not find annotation handler for player {self._model.game.last_player.id}. This should not happen. Error: {error}"
            )

        self._model.annotations.clear()

    def _on_game_started(self, _: events.GameStarted) -> None:
        if self._output_dir is None:
            logging.warning("No output directory set for annotation collection.")
            return

        if self._model.game is None:
            return

        self.annotation_view.clear()
        self._model.annotations.clear()

        for player in self._model.players:
            leg_info = LegAnnotation(
                leg_id=f"{player.name}_{player.id}",
                player_name=player.name,
                starting_score=self._model.game.starting_score,
            )

            self.annotation_handlers[player.id] = LegAnnotationHandler(
                directory=self._output_dir / player.id, info=leg_info
            )

    def _on_game_state_changed(self, _: events.GameStateChanged) -> None:
        # print the annotation from the model
        for annotation, _image in self._model.annotations:
            print(f"Round {annotation.round}:")
            print(annotation.to_json(indent=4))
