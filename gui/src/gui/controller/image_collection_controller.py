import logging

import numpy as np
import ttkbootstrap as ttk
from scored_lib.annotation.image_annotation import ImageAnnotation
from scored_lib.annotation.throw_annotation import AnnotatedThrow

import gui.events.event_types as events
from gui.events.event_channel import EventChannel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.app_view import AppView


class ImageCollectionController(BaseController):
    """
    Collect frames used for annotating
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)
        self._current_frame: np.ndarray | None = None

    def bind_menu(self, _: ttk.Menu) -> None:
        pass

    def bind_components(self) -> None:
        pass

    def start(self) -> None:
        self._event_channel.subscribe(events.FrameCapturedEvent, self._save_frame)
        self._event_channel.subscribe(events.ThrowRecorded, self._on_throw_captured)
        self._event_channel.subscribe(events.ThrowEdited, self._on_throw_edited)
        self._event_channel.subscribe(events.ThrowRemoved, self._on_throw_removed)

    def _save_frame(self, event: events.FrameCapturedEvent) -> None:
        if event.frame is not None:
            self._current_frame = event.frame

    def _on_throw_captured(self, event: events.ThrowRecorded) -> None:
        if self._current_frame is not None and self._model.game is not None:
            num_throws_this_round = len(
                self._model.game.current_leg.current_turn_throws
            )
            if num_throws_this_round == 0:
                num_annotations = len(self._model.annotations)
                assert num_annotations == 0, (
                    "There should be no annotations if there are no throws."
                )

            if self._model.annotations:
                previous_annotation, _ = self._model.annotations[-1]
                previous_annotated_throws = [
                    throw.make_copy() for throw in previous_annotation.throws
                ]
            else:
                previous_annotated_throws = []

            current_throws = previous_annotated_throws + [
                AnnotatedThrow(throw_data=event.result.dart_throw)
            ]
            annotation = ImageAnnotation(
                round=event.result.round,
                is_bust=event.result.bust,
                throw=event.result.throw,
                throws=current_throws,
            )
            logging.debug(
                f"Adding annotation for round {annotation.round}, throw {annotation.throw} with {len(annotation.throws)} throws. {len(previous_annotated_throws)} previous throws."
            )

            self._model.annotations.append((annotation, self._current_frame))
            self._current_frame = None

    def _on_throw_edited(self, event: events.ThrowEdited) -> None:
        if self._model.annotations and len(self._model.annotations) > 0:
            throw_number = event.result.throw
            annotation, _ = self._model.annotations[throw_number - 1]

            annotation.throws[throw_number - 1] = AnnotatedThrow(
                throw_data=event.result.dart_throw,
                keypoints=annotation.throws[throw_number - 1].keypoints,
            )

            logging.debug(
                f"Edited annotation for round {annotation.round}, throw {annotation.throw}. Updated throw data to {event.result.dart_throw}."
            )

    def _on_throw_removed(self, _: events.ThrowRemoved) -> None:
        if self._model.annotations and len(self._model.annotations) > 0:
            self._model.annotations.pop()
            logging.debug(
                f"Removed last annotation. Remaining annotations: {len(self._model.annotations)}."
            )
